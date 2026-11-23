"""Ponto de entrada único para a interface web de Aprendizagem Federada via gRPC."""

import argparse
import socket
import threading
import time
import webbrowser

import uvicorn


def _find_available_port(host: str, preferred: int) -> int:
    for candidate in [preferred, *range(preferred + 1, preferred + 50)]:
        # Verifica primeiro se já existe algum serviço escutando nessa porta
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.15)
            if probe.connect_ex((host, candidate)) == 0:
                continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _open_browser_later(url: str) -> None:
    time.sleep(1.5)
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inicia a interface web interativa para o sistema de Aprendizagem Federada com gRPC."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Endereço de bind (padrão: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Porta HTTP preferencial da interface (padrão: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente")
    args = parser.parse_args()

    port = _find_available_port(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print("=" * 64, flush=True)
    print("  FedIris gRPC — Painel de Aprendizagem Federada (Trabalho 1)", flush=True)
    if port != args.port:
        print(f"  Porta {args.port} ocupada; utilizando automaticamente a porta {port}.", flush=True)
    print(f"  Interface disponível em: {url}", flush=True)
    print("=" * 64, flush=True)

    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    uvicorn.run("app.server:app", host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
