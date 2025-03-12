import ftd2xx
from ftd2xx import FTD2XX

from hassoscopeppy.utils import ensure_list


class Connection:
    def __init__(self, usb: FTD2XX, device_type: str, board_number: int):
        self.device_type = device_type
        self.device_name = str(usb.description, "ASCII")
        self.serial = usb.serial
        self._usb = usb
        self._chunk = 65536
        self.board = board_number

        self._usb.setLatencyTimer(1)  # ms
        self._usb.setTimeouts(
            read = 250,  # read timeout value in milliseconds
            write = 2000  # write timeout value in milliseconds
        )
        usb.setBitMode(0xff, 0x40)
        usb.setUSBParameters(self._chunk * 4, self._chunk * 4)

    def send(self, payload: list[int]) -> int:
        """ Send `payload` data to the device and return number of bytes sent. """
        txlen = 0
        data = bytes(payload)
        for si in range(0, len(data), self._chunk):
            ei = si + self._chunk
            ei = min(ei, len(data))
            chunk = data[si:ei]
            txlen_once = self._usb.write(chunk)
            txlen += txlen_once
            if txlen_once < len(chunk):
                break
        return txlen

    def recv(self, recv_len) -> bytes:
        """ Read `recv_len` bytes. """
        data = b''
        for si in range(0, recv_len, self._chunk):
            ei = si + self._chunk
            ei = min(ei, recv_len)
            chunk_len = ei - si
            chunk = self._usb.read(chunk_len)
            data += chunk
            if len(chunk) < chunk_len:
                break
        return data

    def command(self, payload: list[int]) -> bytes:
        self.send(payload)
        return self.recv(4)


def connect() -> list[Connection]:
    devices = []
    board_number = 1
    for device_serial_number in ensure_list(ftd2xx.listDevices()):
        if device_serial_number.startswith(b'FT'):
            usb: FTD2XX = ftd2xx.openEx(device_serial_number)
            if str(usb.description, "ASCII") == "HaasoscopePro USB2":
                devices.append(Connection(usb, "FTX232H", board_number))
                board_number = board_number + 1
            else:
                usb.close()

    # TODO: Add code link multiple boards together and reorder list of these boards with first one being the one that
    # TODO: will drive clocks of all other boards.

    return devices

if __name__ == '__main__':
    cs = connect()
    print(cs)