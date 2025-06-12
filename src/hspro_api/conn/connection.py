from abc import ABC, abstractmethod


class Connection(ABC):
    def __init__(self, board_number: int, debug: bool):
        self.board = board_number
        self.debug = debug

    @abstractmethod
    def send(self, payload: list[int]) -> int:
        """ Send `payload` data to the device and return number of bytes sent. """

    @abstractmethod
    def recv(self, recv_len: int) -> bytes:
        """ Read `recv_len` bytes. """

    def command(self, payload: list[int], read_response: bool = True) -> bytes | None:
        data = payload + ([0] * (8 - len(payload)))
        self.send(data)
        if self.debug:
            print(f"command :: {data}")
        if read_response:
            return self.recv(4)
        else:
            return None
