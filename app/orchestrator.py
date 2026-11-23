from __future__ import annotations

import copy
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Usa cache local do Hugging Face instantaneamente quando o Iris já foi baixado,
# evitando retentativas HTTP de ~23s em ambientes offline ou com latência.
_HF_IRIS_CACHE = Path.home() / ".cache" / "huggingface" / "datasets" / "hitorilabs___iris"
if _HF_IRIS_CACHE.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import grpc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
START_DIR = PROJECT_ROOT / "start"

if str(START_DIR) not in sys.path:
    sys.path.insert(0, str(START_DIR))

import common_fl
import federated_pb2
import federated_pb2_grpc
import student_fl_client_grpc
import student_fl_server_grpc

SPECIES_FALLBACK = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica",
}

FEATURE_LABELS = {
    "petal_length": "Comprimento da Pétala (cm)",
    "petal_width": "Largura da Pétala (cm)",
    "sepal_length": "Comprimento da Sépala (cm)",
    "sepal_width": "Largura da Sépala (cm)",
}


def format_species_label(raw_label: Any) -> str:
    if isinstance(raw_label, (np.integer, int)):
        return SPECIES_FALLBACK.get(int(raw_label), f"Classe {int(raw_label)}")
    text = str(raw_label).strip()
    if text.isdigit():
        return SPECIES_FALLBACK.get(int(text), f"Classe {text}")
    lower = text.lower()
    if lower in ("setosa", "iris-setosa"):
        return "Iris-setosa"
    if lower in ("versicolor", "iris-versicolor"):
        return "Iris-versicolor"
    if lower in ("virginica", "iris-virginica"):
        return "Iris-virginica"
    return text


def find_free_port(preferred: int = 50051) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def compute_confusion_matrix(
    model_template: Any,
    global_params: List[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model = copy.deepcopy(model_template)
    common_fl.set_model_params(model, global_params)
    y_pred = model.predict(X_test)
    classes = list(model.classes_)
    n = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for truth, pred in zip(y_test, y_pred):
        if truth in class_to_idx and pred in class_to_idx:
            matrix[class_to_idx[truth]][class_to_idx[pred]] += 1
    return {
        "labels": [format_species_label(c) for c in classes],
        "matrix": matrix,
    }


def serialize_weights_snapshot(
    round_num: int,
    params: List[np.ndarray],
    classes: List[Any],
    delta_norm: float = 0.0,
) -> Dict[str, Any]:
    coef = params[0]
    intercept = params[1]
    l2_norm = float(np.sqrt(np.sum(coef ** 2) + np.sum(intercept ** 2)))
    return {
        "round": round_num,
        "features": list(common_fl.FEATURES),
        "classes": [format_species_label(c) for c in classes],
        "coef": [[round(float(v), 5) for v in row] for row in coef.tolist()],
        "intercept": [round(float(v), 5) for v in intercept.tolist()],
        "l2_norm": round(l2_norm, 5),
        "delta_norm": round(float(delta_norm), 5),
    }


@dataclass
class SessionConfig:
    num_clients: int = 5
    total_rounds: int = 10
    local_epochs: int = 1
    poll_interval: float = 0.15
    visual_delay: float = 0.35
    step_mode: bool = False
    port: int = 50051

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "total_rounds": self.total_rounds,
            "local_epochs": self.local_epochs,
            "poll_interval": self.poll_interval,
            "visual_delay": self.visual_delay,
            "step_mode": self.step_mode,
            "port": self.port,
        }


class ObservableFederatedLearningServicer(student_fl_server_grpc.FederatedLearningServicer):
    """
    Estende o FederatedLearningServicer implementado pelo aluno para emitir telemetria
    em tempo real sem alterar nenhuma regra do protocolo gRPC.
    """

    def __init__(self, num_clients: int, total_rounds: int, orchestrator: "FederatedOrchestrator"):
        super().__init__(num_clients=num_clients, total_rounds=total_rounds)
        self._orchestrator = orchestrator

    def GetGlobalModel(self, request, context):
        t0 = time.perf_counter()
        response = super().GetGlobalModel(request, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self.lock:
            received_cids = sorted(int(c) for c in self.received_updates.keys())
        self._orchestrator._record_rpc(
            method="GetGlobalModel",
            cid=int(request.cid),
            round_num=int(response.round),
            accepted=True,
            message=response.message,
            req_bytes=request.ByteSize(),
            resp_bytes=response.ByteSize(),
            latency_ms=elapsed_ms,
            done=bool(response.done),
            received_cids=received_cids,
        )
        return response

    def SubmitUpdate(self, request, context):
        t0 = time.perf_counter()
        ack = super().SubmitUpdate(request, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self.lock:
            received_cids = sorted(int(c) for c in self.received_updates.keys())
        self._orchestrator._record_rpc(
            method="SubmitUpdate",
            cid=int(request.cid),
            round_num=int(request.round),
            accepted=bool(ack.accepted),
            message=ack.message,
            req_bytes=request.ByteSize(),
            resp_bytes=ack.ByteSize(),
            latency_ms=elapsed_ms,
            done=False,
            received_cids=received_cids,
        )
        return ack

    def _consolidate_round_if_ready(self):
        ready = len(self.received_updates) == self.num_clients
        if not ready:
            return

        round_num = self.current_round
        updates_snapshot = {
            cid: {
                "num_examples": int(data["num_examples"]),
                "train_loss": float(data["train_loss"]),
                "train_acc": float(data["train_acc"]),
            }
            for cid, data in self.received_updates.items()
        }
        prev_params = [self.global_params[0].copy(), self.global_params[1].copy()]

        super()._consolidate_round_if_ready()

        # Calcula métricas globais pós-agregação para telemetria da UI
        loss, acc = common_fl.evaluate_global_model(
            self.global_params,
            self.model_template,
            self.X_test_global,
            self.y_test_global,
        )
        confusion = compute_confusion_matrix(
            self.model_template,
            self.global_params,
            self.X_test_global,
            self.y_test_global,
        )
        delta_norm = float(
            np.sqrt(
                np.sum((self.global_params[0] - prev_params[0]) ** 2)
                + np.sum((self.global_params[1] - prev_params[1]) ** 2)
            )
        )
        weights_snap = serialize_weights_snapshot(
            round_num=round_num,
            params=self.global_params,
            classes=list(self.model_template.classes_),
            delta_norm=delta_norm,
        )
        self._orchestrator._on_round_consolidated(
            round_num=round_num,
            total_rounds=self.total_rounds,
            updates_snapshot=updates_snapshot,
            global_loss=loss,
            global_acc=acc,
            confusion=confusion,
            weights_snapshot=weights_snap,
            finished=self.training_finished.is_set(),
        )


class FederatedOrchestrator:
    """Orquestra servidor gRPC real, clientes gRPC reais, baseline manual, inferência e testes."""

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self._subscribers: List[queue.Queue] = []

        self.status: str = "idle"  # idle | starting | running | waiting_step | completed | stopped | error
        self.error_message: Optional[str] = None
        self.config: SessionConfig = SessionConfig()
        self.server_address: Optional[str] = None
        self.current_round: int = 0
        self.active_stage: int = 0  # Etapas 0..5 do PDF
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self._grpc_server: Optional[grpc.Server] = None
        self._servicer: Optional[ObservableFederatedLearningServicer] = None
        self._stop_event = threading.Event()
        self._client_threads: List[threading.Thread] = []

        # Controle de modo passo a passo
        self._step_cond = threading.Condition()
        self._allowed_round: int = 1
        self._waiting_for_step: bool = False

        # Dados de telemetria
        self.clients_state: Dict[int, Dict[str, Any]] = {}
        self.received_cids: List[int] = []
        self.rounds_history: List[Dict[str, Any]] = []
        self.weights_history: List[Dict[str, Any]] = []
        self.confusion_matrix: Optional[Dict[str, Any]] = None
        self.rpc_logs: List[Dict[str, Any]] = []
        self.rpc_stats: Dict[str, Any] = {
            "total_rpcs": 0,
            "get_model_calls": 0,
            "submit_accepted": 0,
            "submit_rejected": 0,
            "total_bytes": 0,
        }
        self.manual_baseline: Optional[Dict[str, Any]] = None
        self.last_test_results: Optional[Dict[str, Any]] = None

        # Cache do modelo mais recente para inferência mesmo após parar a sessão
        self._model_template: Optional[Any] = None
        self._latest_global_params: Optional[List[np.ndarray]] = None
        self._partition_cache: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # SSE Pub/Sub
    # ------------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._state_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._state_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        message = {"type": event_type, "timestamp": time.time(), "data": payload}
        with self._state_lock:
            dead: List[queue.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(message)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    # ------------------------------------------------------------------
    # Snapshot completo do estado
    # ------------------------------------------------------------------
    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            elapsed_sec = None
            if self.started_at is not None:
                end = self.finished_at if self.finished_at is not None else time.time()
                elapsed_sec = round(max(0.0, end - self.started_at), 2)

            return {
                "status": self.status,
                "error_message": self.error_message,
                "config": self.config.to_dict(),
                "server_address": self.server_address,
                "current_round": self.current_round,
                "active_stage": self.active_stage,
                "waiting_for_step": self._waiting_for_step,
                "allowed_round": self._allowed_round,
                "received_cids": list(self.received_cids),
                "elapsed_sec": elapsed_sec,
                "clients": [self.clients_state[k] for k in sorted(self.clients_state.keys())],
                "rounds_history": list(self.rounds_history),
                "weights_history": list(self.weights_history),
                "confusion_matrix": self.confusion_matrix,
                "rpc_stats": dict(self.rpc_stats),
                "rpc_logs": list(self.rpc_logs[-120:]),
                "manual_baseline": self.manual_baseline,
                "has_model": self._model_template is not None and self._latest_global_params is not None,
                "last_test_results": self.last_test_results,
            }

    # ------------------------------------------------------------------
    # Carregamento e metadados de partições
    # ------------------------------------------------------------------
    def _load_partitions_info(self, num_clients: int) -> Tuple[List[Dict[str, Any]], List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        fds = common_fl.build_federated_dataset(num_clients)
        raw_partitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        info_list: List[Dict[str, Any]] = []
        total_train = 0

        for cid in range(num_clients):
            df = fds.load_partition(cid, "train").with_format("pandas")[:]
            X = df[common_fl.FEATURES].to_numpy(dtype=np.float64)
            y = df["species"].to_numpy()
            split_idx = int(0.8 * len(X))
            if split_idx <= 0:
                split_idx = 1
            if split_idx >= len(X):
                split_idx = len(X) - 1
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            raw_partitions.append((X_train, y_train, X_test, y_test))
            total_train += len(X_train)

            train_dist: Dict[str, int] = {}
            for lbl in y_train:
                name = format_species_label(lbl)
                train_dist[name] = train_dist.get(name, 0) + 1

            test_dist: Dict[str, int] = {}
            for lbl in y_test:
                name = format_species_label(lbl)
                test_dist[name] = test_dist.get(name, 0) + 1

            info_list.append(
                {
                    "cid": cid,
                    "train_examples": int(len(X_train)),
                    "test_examples": int(len(X_test)),
                    "train_distribution": train_dist,
                    "test_distribution": test_dist,
                }
            )

        for item in info_list:
            item["fedavg_weight"] = round(item["train_examples"] / max(1, total_train), 4)

        return info_list, raw_partitions

    # ------------------------------------------------------------------
    # Ciclo de vida da sessão gRPC
    # ------------------------------------------------------------------
    def start_session(self, config: SessionConfig) -> Dict[str, Any]:
        self.stop_session()

        with self._state_lock:
            self.config = config
            self.status = "starting"
            self.error_message = None
            self.current_round = 1
            self.active_stage = 0
            self.started_at = time.time()
            self.finished_at = None
            self.rounds_history = []
            self.weights_history = []
            self.confusion_matrix = None
            self.rpc_logs = []
            self.rpc_stats = {
                "total_rpcs": 0,
                "get_model_calls": 0,
                "submit_accepted": 0,
                "submit_rejected": 0,
                "total_bytes": 0,
            }
            self.clients_state = {}
            self.received_cids = []
            self._stop_event.clear()

        with self._step_cond:
            self._allowed_round = 1
            self._waiting_for_step = False

        self._emit("state", self.get_state_snapshot())

        try:
            partitions_info, raw_partitions = self._load_partitions_info(config.num_clients)
            port = find_free_port(config.port)
            address = f"127.0.0.1:{port}"

            servicer = ObservableFederatedLearningServicer(
                num_clients=config.num_clients,
                total_rounds=config.total_rounds,
                orchestrator=self,
            )
            grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max(10, config.num_clients + 4)))
            federated_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, grpc_server)
            grpc_server.add_insecure_port(address)
            grpc_server.start()

            init_loss, init_acc = common_fl.evaluate_global_model(
                servicer.global_params,
                servicer.model_template,
                servicer.X_test_global,
                servicer.y_test_global,
            )
            init_conf = compute_confusion_matrix(
                servicer.model_template,
                servicer.global_params,
                servicer.X_test_global,
                servicer.y_test_global,
            )
            init_weights = serialize_weights_snapshot(
                round_num=0,
                params=servicer.global_params,
                classes=list(servicer.model_template.classes_),
                delta_norm=0.0,
            )

            with self._state_lock:
                self.config.port = port
                self.server_address = address
                self._servicer = servicer
                self._grpc_server = grpc_server
                self._model_template = copy.deepcopy(servicer.model_template)
                self._latest_global_params = [
                    servicer.global_params[0].copy(),
                    servicer.global_params[1].copy(),
                ]
                self.confusion_matrix = init_conf
                self.weights_history = [init_weights]
                self.rounds_history = [
                    {
                        "round": 0,
                        "avg_train_loss": None,
                        "avg_train_acc": None,
                        "global_loss": round(init_loss, 5),
                        "global_acc": round(init_acc, 5),
                        "clients": {},
                    }
                ]
                for pinfo in partitions_info:
                    cid = pinfo["cid"]
                    self.clients_state[cid] = {
                        **pinfo,
                        "status": "idle",
                        "completed_round": 0,
                        "last_loss": None,
                        "last_acc": None,
                        "history": [],
                    }
                self.status = "running"
                self.active_stage = 1

            self._emit("state", self.get_state_snapshot())

            # Inicia os N clientes gRPC reais em threads separadas
            threads: List[threading.Thread] = []
            for cid in range(config.num_clients):
                X_train, y_train, _, _ = raw_partitions[cid]
                t = threading.Thread(
                    target=self._run_monitored_client,
                    args=(cid, address, X_train, y_train, copy.deepcopy(servicer.model_template), config),
                    daemon=True,
                    name=f"fl-client-{cid}",
                )
                t.start()
                threads.append(t)

            with self._state_lock:
                self._client_threads = threads

            return self.get_state_snapshot()
        except Exception as exc:
            with self._state_lock:
                self.status = "error"
                self.error_message = str(exc)
            self._emit("state", self.get_state_snapshot())
            raise

    def advance_step(self) -> Dict[str, Any]:
        with self._step_cond:
            self._allowed_round += 1
            self._waiting_for_step = False
            with self._state_lock:
                if self.status == "waiting_step":
                    self.status = "running"
                    self.active_stage = 1
            self._step_cond.notify_all()
        snapshot = self.get_state_snapshot()
        self._emit("state", snapshot)
        return snapshot

    def stop_session(self) -> Dict[str, Any]:
        self._stop_event.set()
        with self._step_cond:
            self._allowed_round = 10**6
            self._waiting_for_step = False
            self._step_cond.notify_all()

        threads = list(self._client_threads)
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.5)

        if self._grpc_server is not None:
            try:
                self._grpc_server.stop(grace=0.5).wait()
            except Exception:
                pass

        with self._state_lock:
            self._grpc_server = None
            self._servicer = None
            self._client_threads = []
            self.received_cids = []
            if self.status in ("running", "starting", "waiting_step"):
                self.status = "stopped"
                self.finished_at = time.time()

        snapshot = self.get_state_snapshot()
        self._emit("state", snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Cliente gRPC real monitorado (usa exatamente as funções do aluno)
    # ------------------------------------------------------------------
    def _update_client_state(self, cid: int, **changes: Any) -> None:
        with self._state_lock:
            if cid in self.clients_state:
                self.clients_state[cid].update(changes)
        self._emit("client_update", {"cid": cid, **changes})

    def _run_monitored_client(
        self,
        cid: int,
        server_address: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_template: Any,
        config: SessionConfig,
    ) -> None:
        completed_round = 0
        # Pequeno escalonamento visual para que as animações dos clientes fiquem nítidas
        stagger = min(0.18, config.visual_delay * 0.25) * (cid % 5)
        if stagger > 0 and not self._stop_event.wait(stagger):
            pass

        with grpc.insecure_channel(server_address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)

            while not self._stop_event.is_set():
                # No modo passo a passo, aguarda autorização antes de iniciar a próxima rodada
                if config.step_mode:
                    with self._step_cond:
                        while not self._stop_event.is_set() and (completed_round + 1) > self._allowed_round:
                            self._step_cond.wait(timeout=0.2)
                    if self._stop_event.is_set():
                        break

                self._update_client_state(cid, status="polling")
                try:
                    global_model = stub.GetGlobalModel(federated_pb2.ClientHello(cid=cid))
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.UNAVAILABLE:
                        self._update_client_state(cid, status="done")
                        break
                    self._update_client_state(cid, status="error")
                    break

                if global_model.done:
                    self._update_client_state(cid, status="done")
                    break

                if student_fl_client_grpc.should_wait(global_model.round, completed_round):
                    self._update_client_state(cid, status="waiting")
                    self._stop_event.wait(max(0.05, config.poll_interval))
                    continue

                if config.visual_delay > 0:
                    if self._stop_event.wait(config.visual_delay * 0.4):
                        break

                # Etapa 2: Treino local usando funções exatas do aluno
                self._update_client_state(cid, status="training")
                with self._state_lock:
                    self.active_stage = 2
                global_params = student_fl_client_grpc.params_from_proto(global_model.model)
                train_result = common_fl.local_train(
                    X_train=X_train,
                    y_train=y_train,
                    global_params=global_params,
                    model_template=model_template,
                    local_epochs=config.local_epochs,
                )

                if config.visual_delay > 0:
                    if self._stop_event.wait(config.visual_delay * 0.6):
                        break

                # Etapa 3: Serialização e envio via SubmitUpdate
                self._update_client_state(cid, status="sending")
                with self._state_lock:
                    self.active_stage = 3
                update = student_fl_client_grpc.build_client_update(cid, global_model.round, train_result)

                try:
                    ack = stub.SubmitUpdate(update)
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.UNAVAILABLE:
                        self._update_client_state(cid, status="done")
                        break
                    self._update_client_state(cid, status="error")
                    break

                if ack.accepted:
                    completed_round = int(global_model.round)
                    with self._state_lock:
                        c = self.clients_state.get(cid)
                        if c is not None:
                            c["completed_round"] = completed_round
                            c["last_loss"] = round(float(train_result["train_loss"]), 4)
                            c["last_acc"] = round(float(train_result["train_acc"]), 4)
                            c["status"] = "waiting"
                            c["history"].append(
                                {
                                    "round": completed_round,
                                    "loss": round(float(train_result["train_loss"]), 4),
                                    "acc": round(float(train_result["train_acc"]), 4),
                                }
                            )
                    self._emit("client_update", self.clients_state[cid])

                self._stop_event.wait(max(0.05, config.poll_interval))

    # ------------------------------------------------------------------
    # Callbacks chamados pelo ObservableFederatedLearningServicer
    # ------------------------------------------------------------------
    def _record_rpc(
        self,
        method: str,
        cid: int,
        round_num: int,
        accepted: bool,
        message: str,
        req_bytes: int,
        resp_bytes: int,
        latency_ms: float,
        done: bool,
        received_cids: List[int],
    ) -> None:
        entry = {
            "id": 0,
            "time": time.strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "method": method,
            "direction": "server->client" if method == "GetGlobalModel" else "client->server",
            "cid": cid,
            "round": round_num,
            "accepted": accepted,
            "done": done,
            "message": message,
            "req_bytes": req_bytes,
            "resp_bytes": resp_bytes,
            "latency_ms": round(latency_ms, 2),
        }
        with self._state_lock:
            self.rpc_stats["total_rpcs"] += 1
            entry["id"] = self.rpc_stats["total_rpcs"]
            self.rpc_stats["total_bytes"] += req_bytes + resp_bytes
            if method == "GetGlobalModel":
                self.rpc_stats["get_model_calls"] += 1
            elif accepted:
                self.rpc_stats["submit_accepted"] += 1
            else:
                self.rpc_stats["submit_rejected"] += 1

            self.rpc_logs.append(entry)
            if len(self.rpc_logs) > 250:
                self.rpc_logs = self.rpc_logs[-250:]
            self.received_cids = list(received_cids)

        self._emit(
            "rpc",
            {
                "entry": entry,
                "stats": dict(self.rpc_stats),
                "received_cids": list(received_cids),
            },
        )

    def _on_round_consolidated(
        self,
        round_num: int,
        total_rounds: int,
        updates_snapshot: Dict[int, Dict[str, Any]],
        global_loss: float,
        global_acc: float,
        confusion: Dict[str, Any],
        weights_snapshot: Dict[str, Any],
        finished: bool,
    ) -> None:
        avg_train_loss = sum(u["train_loss"] for u in updates_snapshot.values()) / max(1, len(updates_snapshot))
        avg_train_acc = sum(u["train_acc"] for u in updates_snapshot.values()) / max(1, len(updates_snapshot))

        round_record = {
            "round": round_num,
            "avg_train_loss": round(avg_train_loss, 5),
            "avg_train_acc": round(avg_train_acc, 5),
            "global_loss": round(global_loss, 5),
            "global_acc": round(global_acc, 5),
            "clients": {
                str(cid): {
                    "num_examples": data["num_examples"],
                    "train_loss": round(data["train_loss"], 4),
                    "train_acc": round(data["train_acc"], 4),
                }
                for cid, data in updates_snapshot.items()
            },
        }

        with self._state_lock:
            self.active_stage = 4
            self.received_cids = []
            self.rounds_history.append(round_record)
            self.weights_history.append(weights_snapshot)
            self.confusion_matrix = confusion
            if self._servicer is not None:
                self._latest_global_params = [
                    self._servicer.global_params[0].copy(),
                    self._servicer.global_params[1].copy(),
                ]

            if finished:
                self.current_round = total_rounds
                self.status = "completed"
                self.active_stage = 5
                self.finished_at = time.time()
            else:
                self.current_round = round_num + 1
                if self.config.step_mode:
                    self._waiting_for_step = True
                    self.status = "waiting_step"

        if finished:
            with self._step_cond:
                self._allowed_round = total_rounds + 1
                self._waiting_for_step = False
                self._step_cond.notify_all()

        self._emit("state", self.get_state_snapshot())

    # ------------------------------------------------------------------
    # Simulação de RPCs inválidos para demonstrar _validate_update ao vivo
    # ------------------------------------------------------------------
    def inject_validation_rpc(self, kind: str, cid: int = 0) -> Dict[str, Any]:
        with self._state_lock:
            address = self.server_address
            servicer = self._servicer

        if not address or servicer is None:
            raise RuntimeError("Inicie uma sessão gRPC primeiro para enviar RPCs de teste ao servidor.")

        with servicer.lock:
            current_round = servicer.current_round
            proto_model = servicer._params_to_proto(servicer.global_params)
            existing_cids = sorted(int(c) for c in servicer.received_updates.keys())

        if kind == "stale_round":
            target_round = 99
        elif kind == "duplicate":
            target_round = current_round
            target_cid = existing_cids[0] if existing_cids else cid
            with grpc.insecure_channel(address) as channel:
                stub = federated_pb2_grpc.FederatedLearningStub(channel)
                req = federated_pb2.ClientUpdate(
                    cid=target_cid,
                    round=target_round,
                    num_examples=10,
                    train_loss=0.5,
                    train_acc=0.8,
                    model=proto_model,
                )
                first_ack = stub.SubmitUpdate(req)
                if not first_ack.accepted:
                    return {
                        "kind": kind,
                        "accepted": bool(first_ack.accepted),
                        "message": first_ack.message,
                        "server_round": int(first_ack.server_round),
                    }
                second_ack = stub.SubmitUpdate(req)
                return {
                    "kind": kind,
                    "accepted": bool(second_ack.accepted),
                    "message": second_ack.message,
                    "server_round": int(second_ack.server_round),
                }
        elif kind == "after_finish":
            target_round = max(1, current_round - 1)
        else:
            raise ValueError(f"Tipo de teste inválido: {kind}")

        with grpc.insecure_channel(address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)
            req = federated_pb2.ClientUpdate(
                cid=cid,
                round=target_round,
                num_examples=10,
                train_loss=0.5,
                train_acc=0.8,
                model=proto_model,
            )
            ack = stub.SubmitUpdate(req)
            return {
                "kind": kind,
                "accepted": bool(ack.accepted),
                "message": ack.message,
                "server_round": int(ack.server_round),
            }

    # ------------------------------------------------------------------
    # Baseline Manual (equivalente ao fl_sim_manual.py)
    # ------------------------------------------------------------------
    def run_manual_baseline(self, num_clients: int, total_rounds: int, local_epochs: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        _, raw_partitions = self._load_partitions_info(num_clients)
        model_template = common_fl.create_model_template(num_clients)
        global_params = common_fl.get_model_params(model_template)
        X_test_global = np.concatenate([p[2] for p in raw_partitions], axis=0)
        y_test_global = np.concatenate([p[3] for p in raw_partitions], axis=0)

        init_loss, init_acc = common_fl.evaluate_global_model(
            global_params, model_template, X_test_global, y_test_global
        )
        rounds = [
            {
                "round": 0,
                "avg_train_loss": None,
                "avg_train_acc": None,
                "global_loss": round(init_loss, 5),
                "global_acc": round(init_acc, 5),
            }
        ]

        for r in range(1, total_rounds + 1):
            updates = []
            for cid in range(num_clients):
                X_train, y_train, _, _ = raw_partitions[cid]
                res = common_fl.local_train(
                    X_train=X_train,
                    y_train=y_train,
                    global_params=global_params,
                    model_template=model_template,
                    local_epochs=local_epochs,
                )
                updates.append(res)
            global_params = common_fl.aggregate_fedavg(updates)
            loss, acc = common_fl.evaluate_global_model(
                global_params, model_template, X_test_global, y_test_global
            )
            rounds.append(
                {
                    "round": r,
                    "avg_train_loss": round(sum(u["train_loss"] for u in updates) / len(updates), 5),
                    "avg_train_acc": round(sum(u["train_acc"] for u in updates) / len(updates), 5),
                    "global_loss": round(loss, 5),
                    "global_acc": round(acc, 5),
                }
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        baseline = {
            "num_clients": num_clients,
            "total_rounds": total_rounds,
            "local_epochs": local_epochs,
            "elapsed_ms": elapsed_ms,
            "rounds": rounds,
        }
        with self._state_lock:
            self.manual_baseline = baseline
            if self._model_template is None:
                self._model_template = model_template
                self._latest_global_params = global_params
        self._emit("state", self.get_state_snapshot())
        return baseline

    # ------------------------------------------------------------------
    # Inferência Interativa Iris no Modelo Global Atual
    # ------------------------------------------------------------------
    def predict_iris(
        self,
        petal_length: float,
        petal_width: float,
        sepal_length: float,
        sepal_width: float,
    ) -> Dict[str, Any]:
        with self._state_lock:
            if self._model_template is None or self._latest_global_params is None:
                # Carrega um modelo inicial zerado para permitir testar antes do treino
                self._model_template = common_fl.create_model_template(self.config.num_clients)
                self._latest_global_params = common_fl.get_model_params(self._model_template)
            model = copy.deepcopy(self._model_template)
            params = [self._latest_global_params[0].copy(), self._latest_global_params[1].copy()]
            model_round = self.rounds_history[-1]["round"] if self.rounds_history else 0

        common_fl.set_model_params(model, params)
        sample = np.array([[petal_length, petal_width, sepal_length, sepal_width]], dtype=np.float64)
        probs = model.predict_proba(sample)[0]
        pred_idx = int(np.argmax(probs))
        classes = [format_species_label(c) for c in model.classes_]

        return {
            "model_round": model_round,
            "predicted_class": classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": [
                {"species": cls, "probability": round(float(p), 4)}
                for cls, p in zip(classes, probs)
            ],
        }

    # ------------------------------------------------------------------
    # Runner de Testes Unitários (start/tests/)
    # ------------------------------------------------------------------
    def run_unit_tests(self) -> Dict[str, Any]:
        test_modules = [
            "tests.test_01_client_serialization",
            "tests.test_03_server_serialization_and_get",
            "tests.test_04_server_validation",
            "tests.test_05_server_submit_and_consolidate",
            "tests.test_02_client_loop",
        ]

        runner_code = (
            "import json, time, unittest\n"
            f"modules = {json.dumps(test_modules)}\n"
            "loader = unittest.TestLoader()\n"
            "results = []\n"
            "def collect_cases(s, out):\n"
            "    for item in s:\n"
            "        if isinstance(item, unittest.TestSuite):\n"
            "            collect_cases(item, out)\n"
            "        elif item is not None:\n"
            "            out.append((str(item), item._testMethodName))\n"
            "for mod_name in modules:\n"
            "    suite = loader.loadTestsFromName(mod_name)\n"
            "    planned = []\n"
            "    collect_cases(suite, planned)\n"
            "    t0 = time.perf_counter()\n"
            "    res = unittest.TestResult()\n"
            "    suite.run(res)\n"
            "    dt = round((time.perf_counter() - t0) * 1000, 1)\n"
            "    failed_map = {str(t): tb for t, tb in res.failures + res.errors}\n"
            "    cases = [{'name': name, 'passed': tid not in failed_map, 'detail': failed_map.get(tid, '')} for tid, name in planned]\n"
            "    results.append({'module': mod_name, 'ran': res.testsRun, 'passed': res.wasSuccessful(), 'duration_ms': dt, 'cases': cases})\n"
            "print(json.dumps(results))\n"
        )

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            cwd=str(START_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        if proc.returncode != 0:
            payload = {
                "success": False,
                "total_ms": total_ms,
                "total_cases": 0,
                "passed_cases": 0,
                "modules": [],
                "stderr": proc.stderr or proc.stdout,
            }
        else:
            # A última linha do stdout é o JSON (ignora eventuais prints dos métodos testados)
            lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
            parsed = json.loads(lines[-1])
            total_cases = sum(m["ran"] for m in parsed)
            passed_cases = sum(sum(1 for c in m["cases"] if c["passed"]) for m in parsed)
            payload = {
                "success": all(m["passed"] for m in parsed),
                "total_ms": total_ms,
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "modules": parsed,
                "stderr": proc.stderr,
            }

        with self._state_lock:
            self.last_test_results = payload
        self._emit("tests", payload)
        return payload
