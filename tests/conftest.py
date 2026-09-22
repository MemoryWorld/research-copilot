import socket
import threading

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempted = []
    local = threading.local()
    original_pair, original_connect = socket.socketpair, socket.socket.connect

    def pair(*args, **kwargs):
        local.pair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            local.pair = False

    def blocked(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Tests must use mock transports, not network access")

    def connect(sock, address):
        if getattr(local, "pair", False) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return blocked()

    monkeypatch.setattr(socket, "socketpair", pair)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield
    assert not attempted, "An external connection was attempted, even if caught by application code"
