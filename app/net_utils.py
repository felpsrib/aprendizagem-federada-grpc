"""Utilitários de rede seguros para detecção e bind exclusivo de portas (IPv4 + IPv6).

Evita colisões de porta no Windows causadas por SO_REUSEADDR, dual-stack IPv6 (::1)
ou race conditions (TOCTOU) entre checar a porta e iniciar o servidor.
"""

from __future__ import annotations

import socket
from typing import Tuple


def _can_connect(family: int, address: Tuple[str, int], timeout: float = 0.12) -> bool:
    """Retorna True se algum processo já estiver escutando em `address`."""
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(timeout)
            return probe.connect_ex(address) == 0
    except OSError:
        return False


def _is_ipv6_port_busy(port: int) -> bool:
    if not socket.has_ipv6:
        return False
    if _can_connect(socket.AF_INET6, ("::1", port)):
        return True
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock6:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock6.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock6.bind(("::1", port))
    except OSError as exc:
        # Ignora apenas se IPv6 não estiver disponível na interface do SO (EADDRNOTAVAIL/EAFNOSUPPORT)
        if exc.errno in (10049, 10047, 99, 97):
            return False
        return True
    return False


def is_port_occupied(host: str, port: int) -> bool:
    """Verifica se a porta está ocupada em IPv4 ou IPv6 loopback/wildcard."""
    ipv4_target = "127.0.0.1" if host in ("0.0.0.0", "localhost", "") else host
    if _can_connect(socket.AF_INET, (ipv4_target, port)):
        return True

    # No Windows, outro serviço pode estar ouvindo apenas em IPv6 (::1)
    # e interceptar requisições de navegadores que resolvem localhost -> ::1.
    if host in ("127.0.0.1", "localhost", "0.0.0.0", ""):
        if _is_ipv6_port_busy(port):
            return True

    # Testa bind exclusivo em IPv4 para detectar serviços ligados que não aceitam conexão imediata
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind((host, port))
    except OSError:
        return True

    return False


def bind_exclusive_socket(
    host: str,
    preferred_port: int,
    *,
    strict: bool = False,
    search_range: int = 50,
) -> Tuple[socket.socket, int]:
    """Faz bind exclusivo e retorna o socket já aberto para o servidor (sem TOCTOU)."""
    candidates = [preferred_port] if strict else [preferred_port, *range(preferred_port + 1, preferred_port + search_range + 1)]

    for candidate in candidates:
        ipv4_target = "127.0.0.1" if host in ("0.0.0.0", "localhost", "") else host
        if _can_connect(socket.AF_INET, (ipv4_target, candidate)):
            continue
        if host in ("127.0.0.1", "localhost", "0.0.0.0", "") and _is_ipv6_port_busy(candidate):
            continue

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, candidate))
            sock.listen(128)
            sock.set_inheritable(True)
            return sock, int(sock.getsockname()[1])
        except OSError:
            sock.close()
            continue

    if strict:
        raise OSError(
            f"A porta {preferred_port} em {host} já está em uso por outro processo."
        )

    # Fallback final: deixa o SO escolher uma porta efêmera garantidamente livre
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind((host, 0))
    sock.listen(128)
    sock.set_inheritable(True)
    return sock, int(sock.getsockname()[1])


def find_free_port(host: str = "127.0.0.1", preferred: int = 50051, search_range: int = 50) -> int:
    """Encontra uma porta livre verificando IPv4, IPv6 e bind exclusivo."""
    for candidate in [preferred, *range(preferred + 1, preferred + search_range + 1)]:
        if not is_port_occupied(host, candidate):
            return candidate

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
