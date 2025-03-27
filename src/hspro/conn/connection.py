from abc import ABC, abstractmethod


class Connection(ABC):
    def __init__(self, board_number: int):
        self.board = board_number

    @abstractmethod
    def send(self, payload: list[int]) -> int:
        """ Send `payload` data to the device and return number of bytes sent. """

    @abstractmethod
    def recv(self, recv_len: int) -> bytes:
        """ Read `recv_len` bytes. """

    def command(self, payload: list[int]) -> bytes:
        data = payload + ([0] * (8 - len(payload)))
        self.send(data)
        print(f"command :: {data}")
        return self.recv(4)
