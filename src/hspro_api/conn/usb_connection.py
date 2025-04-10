from ftd2xx import FTD2XX

from hspro_api.conn.connection import Connection


class USBConnection(Connection):
    def __init__(self, usb: FTD2XX, device_type: str, board_number: int, debug: bool):
        super().__init__(board_number, debug)
        self.device_type = device_type
        self.device_name = str(usb.description, "ASCII")
        self.serial = usb.serial
        self._usb = usb
        self._chunk = 65536

        self._usb.setLatencyTimer(1)  # ms
        self._usb.setTimeouts(
            read=250,  # read timeout value in milliseconds
            write=2000  # write timeout value in milliseconds
        )
        usb.setBitMode(0xff, 0x40)
        usb.setUSBParameters(self._chunk * 4, self._chunk * 4)

    def send(self, payload: list[int]) -> int:
        """ Send `payload` data to the device and return number of bytes sent. """
        if self.debug:
            print("usb::send", ",".join(f"{b}" for b in payload))
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
            av = self._usb.getQueueStatus()
            chunk = self._usb.read(chunk_len)
            data += chunk
            if len(chunk) < chunk_len:
                break
        return data
