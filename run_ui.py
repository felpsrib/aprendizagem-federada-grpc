"""Ponto de entrada único para a interface web de Aprendizagem Federada via gRPC."""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn

from app.net_utils import bind_exclusive_socket
from app.server import APP_IDENTIFIER, INSTANCE_ID, app


def _open_browser_when_ready(url: str, expected_instance_id: str, stop_event: threading.Event) -> None:
    """Só abre o navegador após confirmar que ESTE servidor FedIris está respondendo na porta."""
    health_url = f"{url}/api/health"
    deadline = time.monotonic() + 20.0

    while not stop_event.is_set() and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=0.6) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("app") == APP_IDENTIFIER and data.get("instance_id") == expected_instance_id:
                        webbrowser.open(url)
                        return
                    # Outro sistema respondeu inesperadamente nessa URL — não abrir!
                    print(
                        f"[Aviso] Outro serviço respondeu em {url}. O navegador não foi aberto automaticamente.",
                        flush=True,
                    )
                    return
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
        stop_event.wait(0.15)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inicia a interface web interativa para o sistema de Aprendizagem Federada com gRPC."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Endereço de bind (padrão: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Porta HTTP preferencial da interface (padrão: 8000)")
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Encerra com erro se a porta escolhida já estiver em uso, em vez de buscar outra porta livre",
    )
    parser.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente")
    args = parser.parse_args()

    try:
        bound_socket, port = bind_exclusive_socket(
            args.host,
            args.port,
            strict=args.strict_port,
        )
    except OSError as exc:
        print(f"[Erro] Não foi possível reservar a porta {args.port} em {args.host}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    url = f"http://{args.host}:{port}"
    print("=" * 64, flush=True)
    print("  FedIris gRPC — Painel de Aprendizagem Federada (Trabalho 1)", flush=True)
    if port != args.port:
        print(
            f"  [Aviso] Porta {args.port} já está em uso por outro processo.",
            flush=True,
        )
        print(
            f"  [Aviso] Utilizando automaticamente a porta exclusiva {port}.",
            flush=True,
        )
    print(f"  Interface disponível em: {url}", flush=True)
    print("=" * 64, flush=True)

    stop_event = threading.Event()
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, INSTANCE_ID, stop_event),
            daemon=True,
        ).start()

    config = uvicorn.Config(app, host=args.host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[bound_socket])
    finally:
        stop_event.set()
        bound_socket.close()


if __name__ == "__main__":
    main()
