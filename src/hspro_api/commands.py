import math
import time
from enum import Enum

import bitstruct

from hspro_api.conn.connection import Connection
from hspro_api.registers_enum import RegisterIndex
from hspro_api.utils import int_to_bytes, getbit


class TriggerType(Enum):
    DISABLED = 0
    ON_RISING_EDGE = 1
    ON_FALLING_EDGE = 2
    EXTERNAL = 3
    AUTO = 4


class Commands:
    def __init__(self, connection: Connection):
        self.conn = connection
        self.debug: bool = False

    def get_waveform_data(self, num_bytes: int) -> bytes:
        # self.conn._usb.purge()
        self.conn.send([0, 0, 0, 0] + list(num_bytes.to_bytes(4, byteorder="little")))  # send the 8 bytes to usb
        return self.conn.recv(num_bytes)

    def arm_trigger(
            self,
            trigger_type: TriggerType,
            two_channels: bool,
            oversample: bool,
            samples_after_trigger: int
    ) -> tuple[bool, int]:
        trigger_data = self.conn.command(
            [
                1, trigger_type.value,
                (1 if two_channels else 0) + 2 * (1 if oversample else 0),
                0
            ] + int_to_bytes(samples_after_trigger)  # length to take after trigger (bytes 4 and 5)
        )
        acqstate = trigger_data[0]
        if acqstate == 251:  # an event is ready to be read out
            # print("board",board,"sample triggered", binprint(triggercounter[3]), binprint(triggercounter[2]), binprint(triggercounter[1]))
            gotzerobit = False
            sample_triggered = 0
            for s in range(21):
                thebit = getbit(trigger_data[int(s / 8) + 1], s % 8)
                if thebit == 0:
                    gotzerobit = True
                elif thebit == 1 and gotzerobit:
                    sample_triggered = s
                    gotzerobit = False
            return True, sample_triggered
        else:
            return False, -1

    def get_version(self) -> int:
        """ Return firmware version. """
        version = self.read_register(RegisterIndex.version)
        if self.debug:
            print(f"version = {version}")
        return version
        # return int.from_bytes(self.conn.command([2, 0, 0, 0, 0, 0, 0, 0]), byteorder="little")

    def get_boardin(self) -> int:
        """ TODO: Describe me """
        res = self.conn.command([2, 1, 0, 0, 0, 0, 0, 0])[0]

        if self.debug:
            print(f"Board in bits: {res:08b}")

        return res

    def get_overrange_counter(self, c: int) -> int:
        """ TODO: Describe me """
        return int.from_bytes(self.conn.command([2, 2, c, 0, 0, 0, 0, 0]), byteorder="little")

    def get_eventconter(self) -> int:
        """ TODO: Describe me """
        return int.from_bytes(self.conn.command([2, 3, 0, 0, 0, 0, 0, 0]), byteorder="little")

    def auxoutselector(self, val):
        # set aux out SMA on back panel to clklvds (0) or trig out (1)
        self.conn.command([2, 10, val, 0, 99, 99, 99, 99])

    def clkout_ena(self, enable: bool) -> None:
        self.conn.command([2, 9, (1 if enable else 0), 0, 99, 99, 99, 99])  # turn on/off lvdsout_clk

    def get_downsample_merging_counter_triggered(self) -> tuple[int, int]:
        """ TODO: Describe me """
        results = self.conn.command([2, 4, 0, 0, 0, 0, 0, 0])
        return results[0], results[1]  # downsamplemergingcounter, triggerphase

    def set_lvdsout_spare(self):
        """ TODO: Implement me """
        pass

    def set_fanon(self, enable: bool) -> bool:
        """
        Enables or disable on board fan. Return previous state of fan.
        """
        return self.conn.command([2, 6, (1 if enable else 0), 0, 0, 0, 0, 0])[0] == 1

    def set_prelength_to_take(self, prelength_to_take: int) -> None:
        """ TODO: Describe me """
        if prelength_to_take > 65535:
            raise RuntimeError(f"prelength_to_take value cannot be greater than 65535")
        else:
            self.conn.command([2, 7] + int_to_bytes(prelength_to_take) + [0, 0])
            # self.conn.command([2, 7] + list(prelength_to_take.to_bytes(2, byteorder="little")) + [0, 0, 0, 0])

    def set_rolling(self, rooling_trigger: bool) -> None:
        """ TODO: Describe me """
        self.conn.command([2, 8, (1 if rooling_trigger else 0), 0, 0, 0, 0, 0])

    def spi_command(
            self,
            name: str,
            first: int,
            second: int,
            third: int,
            read: bool,
            fourth: int = 100,
            show_bin: bool = False,
            cs: int = 0,
            nbyte: int = 3,
            quiet=False
    ) -> bytes | None:
        """
        * first byte to send, start of address
        * second byte to send, rest of address
        * third byte to send, value to write, ignored during read
        * cs is which chip to select, adc 0 by default
        * nbyte is 2 or 3, second byte is ignored in case of 2 bytes
        """
        if read: first = first + 0x80  # set the highest bit for read, i.e. add 0x80
        spires = self.conn.command([3, cs, first, second, third, fourth, 100, nbyte])  # get SPI result from command
        return spires if read else None

    def spi_command2(
            self,
            name: str,
            first: int, second: int, third: int, fourth: int,
            read: bool,
            cs: int = 0,
            nbyte: int = 3
    ) -> tuple[bytes, bytes] | None:
        """
        * first byte to send, start of address
        * second byte to send, rest of address
        * third byte to send, value to write, ignored during read, to address +1 (the higher 8 bits)
        * fourth byte to send, value to write, ignored during read
        * cs is which chip to select, adc 0 by default
        * nbyte is 2 or 3, second byte is ignored in case of 2 bytes
        """
        if read: first = first + 0x80  # set the highest bit for read, i.e. add 0x80

        # get SPI result from command
        spires = self.conn.command([3, cs, first, second, fourth, 100, 100, nbyte])

        # get SPI result from command for next byte
        spires2 = self.conn.command([3, cs, first, second + 0x01, third, 100, 100, nbyte])
        return spires, spires2

    def set_spi_mode(self, mode: int) -> None:
        """ Set SPI mode (polarity of clk and data). """
        self.conn.command([4, mode, 0, 0, 0, 0, 0, 0])

    def reset_plls(self) -> None:
        """ TODO: Describe me """
        self.conn.command([5, 0, 0, 0, 0, 0, 0, 0])

    def set_clk_phase_adjust(self, pll_num: int, pll_out_num: int, up_down: bool) -> None:
        """
        TODO: Describe me

        pll_num: can only be 0, 1, 2 or 3.
        """
        self.conn.command([6, pll_num, int(pll_out_num + 2), (1 if up_down else 0), 0, 0, 0, 0])

    def clk_switch(self) -> int:
        """ TODO: Describe me """
        clockinfo = self.conn.command([7, 0, 0, 0, 99, 99, 99, 99])
        return clockinfo[1]

    def set_trigger_props(
            self,
            trigger_level: int,
            trigger_delta: int,
            trigger_pos: int,
            tot: int,
            trigger_on_channel: int
    ) -> None:
        self.conn.command([
            8, trigger_level + 1, trigger_delta, int(trigger_pos / 256), trigger_pos % 256, tot, trigger_on_channel, 0
        ])

    def set_downsample(self, downsample: int, highres: bool, downsample_merging: int) -> None:
        self.conn.command([9, downsample, (1 if highres else 0), downsample_merging, 0, 0, 0, 0])

    def set_boardout(self, control_bit: int, value_bit: int):
        """ TODO: Describe me """
        self.conn.command([10, control_bit, value_bit, 0, 0, 0, 0, 0])

    def set_led(self, rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> None:
        """ TODO: Describe me """
        self.conn.command([11, 1, rgb1[1], rgb1[0], rgb1[2], rgb2[1], rgb2[0], rgb2[2]])  # send
        self.conn.command([11, 0, rgb1[1], rgb1[0], rgb1[2], rgb2[1], rgb2[0], rgb2[2]])  # stop sending

    def is_capture_available(self) -> tuple[bool, int]:
        trigger_data = self.conn.command([12, 0, 0, 0, 0, 0, 0, 0])
        acqstate = trigger_data[0]
        if self.debug:
            print(f"acqstate {acqstate} @ {time.time()}")
        if acqstate == 251:  # an event is ready to be read out
            # print("board",board,"sample triggered", binprint(triggercounter[3]), binprint(triggercounter[2]), binprint(triggercounter[1]))
            gotzerobit = False
            sample_triggered = 0
            for s in range(20):
                thebit = getbit(trigger_data[int(s / 8) + 1], s % 8)
                if thebit == 0:
                    gotzerobit = True
                elif thebit == 1 and gotzerobit:
                    sample_triggered = s
                    gotzerobit = False
            return True, sample_triggered
        else:
            return False, -1

    def force_arm_trigger(
            self,
            trigger_type: TriggerType,
            two_channels: bool,
            oversample: bool,
            absolute_trigger_pos: float,
            expect_samples: int
    ) -> bool:
        samples_after_trigger = expect_samples - absolute_trigger_pos + 1
        trigger_data = self.conn.command(
            [
                13, trigger_type.value,
                (1 if two_channels else 0) + 2 * (1 if oversample else 0),
                0
            ] + int_to_bytes(samples_after_trigger)  # length to take after trigger (bytes 4 and 5)
        )
        if self.debug:
            print("trigger data", trigger_data)
        return trigger_data[0] == 1

    def force_data_acquisition(self) -> int:
        response = self.conn.command([14, 0, 0, 0, 0, 0, 0, 0])
        if self.debug:
            print(f"force_data_acquisition >>> {response[0]}")
        return response[0]

    def set_voltage_div(self, channel: int, dV: float, do_oversample: bool, ten_x_probe: bool) -> float:
        scaling_factor = 0.2 * (10 if ten_x_probe else 1) * (2 if do_oversample else 1)
        db = int(math.log10(scaling_factor / dV) * 20)

        actual_voltage_per_division = scaling_factor / pow(10, db / 20.0)
        self.set_spi_mode(0)

        chan = (channel + 1) % 2 if do_oversample else channel
        if chan == 0: self.spi_command("Amp Gain 0", 0x02, 0x00, 26 - db, False, cs=2, nbyte=2, quiet=True)
        if chan == 1: self.spi_command("Amp Gain 1", 0x02, 0x00, 26 - db, False, cs=1, nbyte=2, quiet=True)
        return actual_voltage_per_division

    def set_offset_V(self, channel: int, offsetV: float, do_oversample: bool, ten_x_probe: bool) -> float:
        # dacval = int((pow(2, 16) - 1) * (val * scaling / 2 + 500) / 1000)
        # dacval = int(((pow(2, 16) - 1) * offsetV / (3 * (2 if do_oversample else 1)) + 500))
        tx = 10 if ten_x_probe else 1
        A = 2 if do_oversample else 1
        dacval = int((pow(2, 16) - 1) * ((offsetV / (3 * tx * A)) + 0.5))
        if dacval < 0:
            dacval = 0
        elif dacval > pow(2, 16):
            dacval = pow(2, 16)

        self.set_spi_mode(1)
        chan = (channel + 1) % 2 if do_oversample else channel

        if chan == 1: self.spi_command("DAC 1 value", 0x18, dacval >> 8, dacval % 256, False, cs=4, quiet=True)
        if chan == 0: self.spi_command("DAC 2 value", 0x19, dacval >> 8, dacval % 256, False, cs=4, quiet=True)
        self.set_spi_mode(0)

        real_offset_V = (dacval / (pow(2, 16) - 1) - 0.5) * 2 * tx * A
        if self.debug:
            print(f"set_offset_V {offsetV} -> {dacval} -> {real_offset_V}")
        return real_offset_V

    def read_register(self, reg: RegisterIndex) -> int:
        response = self.conn.command([14, reg.value, 0, 0, 0, 0, 0, 0])
        if self.debug:
            print(f"read_register {reg} -> {response}")
        match reg:
            case 0:  # ram_preoffset
                _, v = bitstruct.unpack(">u6u10", bytes([response[1], response[0]]))
                return v
            case 1:  # ram_address_triggered_sync
                _, v = bitstruct.unpack(">u6u10", bytes([response[1], response[0]]))
                return v
            case 2:  # spistate
                return response[0]
            case 3:  # version
                return int.from_bytes(response, byteorder="little")
                # return bitstruct.unpack("<u32", response[::-1])[0]
            case 4:  # boardin
                return response[0]
            case 5:  # acqstate
                return response[0]
            case 6:  # eventcounter
                return int.from_bytes(response, byteorder="little")
            case 7:  # sample_triggered
                return int.from_bytes(response, byteorder="little")
            case 8:  # downsamplemergingcounter_triggered
                return response[0]
            case RegisterIndex.downsamplemerging:  # downsamplemerging
                return response[0]
            case RegisterIndex.downsample:  # downsample
                return response[0]
            case RegisterIndex.highres:
                return response[0]
            case RegisterIndex.upperthresh:
                return int.from_bytes(response, byteorder="little")
            case _:
                raise RuntimeError(f"Unknown register index {reg}")

    def set_configured(self, configured: bool) -> None:
        self.conn.command([15, 1 if configured else 0])
