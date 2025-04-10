import ftd2xx
from ftd2xx import FTD2XX

from hspro_api.commands import Commands
from hspro_api.conn.connection import Connection
from hspro_api.conn.usb_connection import USBConnection
from hspro_api.utils import ensure_list


def connect(debug: bool) -> list[Connection]:
    devices = []
    board_number = 1
    for device_serial_number in ensure_list(ftd2xx.listDevices()):
        if device_serial_number.startswith(b'FT'):
            usb: FTD2XX = ftd2xx.openEx(device_serial_number)
            if str(usb.description, "ASCII") == "HaasoscopePro USB2":
                connection = USBConnection(usb, "FTX232H", board_number, debug)

                # Something is off in our board; The first time it is powered on we get
                # incorrect firmware version number one or two time. After that it works fine.
                # For now, workaround is simply to request version several times before returning
                # connection to the user.
                cmd = Commands(connection)
                cmd.get_version()
                cmd.get_version()
                cmd.get_version()

                devices.append(connection)
                board_number = board_number + 1
            else:
                usb.close()

    # TODO: Add code link multiple boards together and reorder list of these boards with first one being the one that
    # TODO: will drive clocks of all other boards.

    return devices
