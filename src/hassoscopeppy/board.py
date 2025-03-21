import time
from enum import IntEnum

from hassoscopeppy.adf435x_core import FeedbackSelect, calculate_regs, DeviceType, BandSelectClockMode, make_regs, \
    PDPolarity, ClkDivMode
from hassoscopeppy.connection import Connection, connect
from hassoscopeppy.utils import bit_asserted, int_to_bytes


class TriggerType(IntEnum):
    ON_RISING_EDGE = 1
    ON_FALLING_EDGE = 2
    EXTERNAL = 3


class Commands:
    READ_BOARD_IN = [2, 1, 0, 0, 0, 0, 0, 0]
    SET_SPI_MODE = lambda mode: [4, mode, 0, 0, 0, 0, 0, 0]
    RESET_PLLS = [5, 99, 99, 99, 100, 100, 100, 100]
    SWITCH_CLOCKS = [7, 0, 0, 0, 99, 99, 99, 99]
    FAN_ON_COMMAND = lambda fan_on: [2, 6, fan_on, 100, 100, 100, 100, 100]

class Command:
    ARM_TRIGGER = 1

class Board:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.sample_rate_GHz = 3.2  # freq in GHz
        self.themuxoutV = True
        self.debug_spi = False
        self.phasecs = []
        self.plljustreset = False
        self.dopattern = 0
        self.dotwochannel = False
        self.dooverrange = False
        self.dooversample = False
        self.num_chan_per_board = 2

        self.expect_samples = 100
        self.expect_samples_extra = 5  # enough to cover downsample shifting and toff shifting
        self.nsubsamples = 10 * 4 + 8 + 2  # extra 4 for clk+str, and 2 dead beef

        self.adf_reset()
        self.pll_reset()
        self.setupboard(self.dopattern, self.dotwochannel, self.dooverrange)
        for c in range(self.num_chan_per_board):
            self.setchanacdc(chan = c, ac = 0, doswap = self.dooversample)

    def get_waveform(self):
        # length to request: each adc bit is stored as 10 bits in 2 bytes, a couple extra for shifting later
        expect_len = (self.expect_samples + self.expect_samples_extra) * 2 * self.nsubsamples

        self.connection.send([0, 99, 99, 99] + int_to_bytes(expect_len))  # send the 4 bytes to usb
        data = self.connection.recv(expect_len)  # recv from usb
        rx_len = len(data)
        # self.total_rx_len += rx_len
        if expect_len != rx_len:
            print('*** expect_len (%d) and rx_len (%d) mismatch' % (expect_len, rx_len))
        return data

    def setchanacdc(self, chan, ac, doswap: bool):
        if doswap: chan = (chan + 1) % 2

        match chan:
            case 0:
                self.connection.command([10, 1, not ac, 0, 0, 0, 0, 0])  # controlbit = 1
            case 1:
                self.connection.command([10, 5, not ac, 0, 0, 0, 0, 0])  # controlbit = 5

    def setfan(self, fanon: int):
        res = self.connection.command(Commands.FAN_ON_COMMAND(fanon))  # set / get fan status
        print(f"Set fan {fanon} and it was {res[0]}")

    def swapinputs(self, doswap: bool, insetup: bool = False):
        if not insetup:
            self.spi_cmd("LVDS_EN", 0x02, 0x00, 0x00, False)  # disable LVDS interface
            self.spi_cmd("CAL_EN", 0x00, 0x61, 0x00, False)  # disable calibration
        if doswap:
            self.spi_cmd("INPUT_MUX", 0x00, 0x60, 0x12, False)  # swap inputs
        else:
            self.spi_cmd("INPUT_MUX", 0x00, 0x60, 0x01, False)  # unswap inputs
        if not insetup:
            self.spi_cmd("CAL_EN", 0x00, 0x61, 0x01, False)  # enable calibration
            self.spi_cmd("LVDS_EN", 0x02, 0x00, 0x01, False)  # enable LVDS interface

    def setupboard(self, dopattern, twochannel, dooverrange):
        self.setfan(1)

        self.set_spi_mode(0)
        self.spi_cmd("DEVICE_CONFIG", 0x00, 0x02, 0x00, False)  # power up
        # spicommand(usb, "DEVICE_CONFIG", 0x00, 0x02, 0x03, False) # power down
        self.spi_cmd2("VENDOR", 0x00, 0x0c, 0x00, 0x00, True)
        self.spi_cmd("LVDS_EN", 0x02, 0x00, 0x00, False)  # disable LVDS interface
        self.spi_cmd("CAL_EN", 0x00, 0x61, 0x00, False)  # disable calibration

        if twochannel:
            self.spi_cmd("LMODE", 0x02, 0x01, 0x03, False)  # LVDS mode: aligned, demux, dual channel, 12-bit
            # spicommand("LMODE", 0x02, 0x01, 0x01, False)  # LVDS mode: staggered, demux, dual channel, 12-bit
        else:
            self.spi_cmd("LMODE", 0x02, 0x01, 0x07, False)  # LVDS mode: aligned, demux, single channel, 12-bit
            # self.spicommand("LMODE", 0x02, 0x01, 0x37, False)  # LVDS mode: aligned, demux, single channel, 8-bit
            # self.spicommand("LMODE", 0x02, 0x01, 0x05, False)  # LVDS mode: staggered, demux, single channel, 12-bit

        self.spi_cmd("LVDS_SWING", 0x00, 0x48, 0x00, False)  # high swing mode
        # self.spicommand("LVDS_SWING", 0x00, 0x48, 0x01, False)  #low swing mode

        self.spi_cmd("LCTRL", 0x02, 0x04, 0x0a, False)  # use LSYNC_N (software), 2's complement
        # self.spicommand("LCTRL", 0x02, 0x04, 0x08, False)  # use LSYNC_N (software), offset binary

        self.swapinputs(doswap = False, insetup = True)

        # self.spicommand("TAD", 0x02, 0xB7, 0x01, False)  # invert clk
        self.spi_cmd("TAD", 0x02, 0xB7, 0x00, False)  # don't invert clk

        tad = 0
        self.spi_cmd("TAD", 0x02, 0xB6, tad, False)  # adjust TAD (time of ADC relative to clk)

        if dooverrange:
            self.spi_cmd("OVR_CFG", 0x02, 0x13, 0x0f, False)  # overrange on
            self.spi_cmd("OVR_T0", 0x02, 0x11, 0xf2, False)  # overrange threshold 0
            self.spi_cmd("OVR_T1", 0x02, 0x12, 0xab, False)  # overrange threshold 1
        else:
            self.spi_cmd("OVR_CFG", 0x02, 0x13, 0x07, False)  # overrange off

        if dopattern:
            self.spi_cmd("PAT_SEL", 0x02, 0x05, 0x11, False)  # test pattern
            usrval = 0x00
            if dopattern == 1:
                self.spi_cmd2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.spi_cmd2("UPAT1", 0x01, 0x82, usrval, usrval + 1, False)  # set pattern sample 1
                self.spi_cmd2("UPAT2", 0x01, 0x84, usrval, usrval + 2, False)  # set pattern sample 2
                self.spi_cmd2("UPAT3", 0x01, 0x86, usrval, usrval + 4, False)  # set pattern sample 3
                self.spi_cmd2("UPAT4", 0x01, 0x88, usrval, usrval + 8, False)  # set pattern sample 4
                self.spi_cmd2("UPAT5", 0x01, 0x8a, usrval, usrval + 16, False)  # set pattern sample 5
                self.spi_cmd2("UPAT6", 0x01, 0x8c, usrval, usrval + 32, False)  # set pattern sample 6
                self.spi_cmd2("UPAT7", 0x01, 0x8e, usrval, usrval + 64, False)  # set pattern sample 7
            if dopattern == 2:
                self.spi_cmd2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.spi_cmd2("UPAT1", 0x01, 0x82, usrval + 1, usrval, False)  # set pattern sample 1
                self.spi_cmd2("UPAT2", 0x01, 0x84, usrval, usrval, False)  # set pattern sample 2
                self.spi_cmd2("UPAT3", 0x01, 0x86, usrval + 2, usrval, False)  # set pattern sample 3
                self.spi_cmd2("UPAT4", 0x01, 0x88, usrval, usrval, False)  # set pattern sample 4
                self.spi_cmd2("UPAT5", 0x01, 0x8a, usrval + 3, usrval, False)  # set pattern sample 5
                self.spi_cmd2("UPAT6", 0x01, 0x8c, usrval, usrval, False)  # set pattern sample 6
                self.spi_cmd2("UPAT7", 0x01, 0x8e, usrval + 4, usrval, False)  # set pattern sample 7
            if dopattern == 3:
                self.spi_cmd2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.spi_cmd2("UPAT1", 0x01, 0x82, usrval + 0x01, usrval + 0x01, False)  # set pattern sample 1
                self.spi_cmd2("UPAT2", 0x01, 0x84, usrval + 0x01, usrval + 0x03, False)  # set pattern sample 2
                self.spi_cmd2("UPAT3", 0x01, 0x86, usrval + 0x03, usrval + 0x07, False)  # set pattern sample 3
                self.spi_cmd2("UPAT4", 0x01, 0x88, usrval + 0x03, usrval + 0x0f, False)  # set pattern sample 4
                self.spi_cmd2("UPAT5", 0x01, 0x8a, usrval + 0x07, usrval + 0x7f, False)  # set pattern sample 5
                self.spi_cmd2("UPAT6", 0x01, 0x8c, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 6
                self.spi_cmd2("UPAT7", 0x01, 0x8e, usrval + 0x08, usrval, False)  # set pattern sample 7
            if dopattern == 4:
                self.spi_cmd2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.spi_cmd2("UPAT1", 0x01, 0x82, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 1
                self.spi_cmd2("UPAT2", 0x01, 0x84, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 2
                self.spi_cmd2("UPAT3", 0x01, 0x86, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 3
                self.spi_cmd2("UPAT4", 0x01, 0x88, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 4
                self.spi_cmd2("UPAT5", 0x01, 0x8a, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 5
                self.spi_cmd2("UPAT6", 0x01, 0x8c, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 6
                self.spi_cmd2("UPAT7", 0x01, 0x8e, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 7
            # self.spicommand("UPAT_CTRL", 0x01, 0x90, 0x0e, False)  # set lane pattern to user, invert a bit of B C D
            self.spi_cmd("UPAT_CTRL", 0x01, 0x90, 0x00, False)  # set lane pattern to user
        else:
            self.spi_cmd("PAT_SEL", 0x02, 0x05, 0x02, False)  # normal ADC data
            self.spi_cmd("UPAT_CTRL", 0x01, 0x90, 0x1e, False)  # set lane pattern to default

        self.spi_cmd("CAL_EN", 0x00, 0x61, 0x01, False)  # enable calibration
        self.spi_cmd("LVDS_EN", 0x02, 0x00, 0x01, False)  # enable LVDS interface
        self.spi_cmd("LSYNC_N", 0x02, 0x03, 0x00, False)  # assert ~sync signal
        self.spi_cmd("LSYNC_N", 0x02, 0x03, 0x01, False)  # deassert ~sync signal
        # self.spicommand("CAL_SOFT_TRIG", 0x00, 0x6c, 0x00, False)
        # self.spicommand("CAL_SOFT_TRIG", 0x00, 0x6c, 0x01, False)

        self.set_spi_mode(0)
        self.spi_cmd("Amp Rev ID", 0x00, 0x00, 0x00, True, cs = 1, nbyte = 2)
        self.spi_cmd("Amp Prod ID", 0x01, 0x00, 0x00, True, cs = 1, nbyte = 2)
        self.spi_cmd("Amp Rev ID", 0x00, 0x00, 0x00, True, cs = 2, nbyte = 2)
        self.spi_cmd("Amp Prod ID", 0x01, 0x00, 0x00, True, cs = 2, nbyte = 2)

        self.set_spi_mode(1)

        self.spi_cmd("DAC ref on", 0x38, 0xff, 0xff, False, cs = 4)
        self.spi_cmd("DAC gain 1", 0x02, 0xff, 0xff, False, cs = 4)

        self.set_spi_mode(0)
        self.do_offset(chan = 0, value = 0, scaling = 1, doswap = False)
        self.do_offset(chan = 1, value = 0, scaling = 1, doswap = False)
        self.set_gain(chan = 0, value = 0, doswap = False)
        self.set_gain(chan = 1, value = 0, doswap = False)

    def set_gain(self, chan: int, value: int, doswap: bool):
        self.set_spi_mode(0)
        # 00 to 20 is 26 to -6 dB, 0x1a is no gain
        if doswap: chan = (chan + 1) % 2

        if chan == 0:
            self.spi_cmd("Amp Gain 0", 0x02, 0x00, 26 - value, False, cs = 2, nbyte = 2, quiet = True)

        if chan == 1:
            self.spi_cmd("Amp Gain 1", 0x02, 0x00, 26 - value, False, cs = 1, nbyte = 2, quiet = True)

    def do_offset(self, chan: int, value: float, scaling: float, doswap: bool) -> bool:
        self.set_spi_mode(1)

        # if doswap: val= -val
        dacval = int((pow(2, 16) - 1) * (value * scaling / 2 + 500) / 1000)
        # print("dacval is", dacval,"and doswap is",doswap,"and val is",val)
        ret = False
        if 0 < dacval < pow(2, 16):
            ret = True
            if doswap: chan = (chan + 1) % 2
            if chan == 1:
                self.spi_cmd("DAC 1 value", 0x18, dacval >> 8, dacval % 256, False, cs = 4, quiet = True)

            if chan == 0:
                self.spi_cmd("DAC 2 value", 0x19, dacval >> 8, dacval % 256, False, cs = 4, quiet = True)

        self.set_spi_mode(0)
        return ret

    def set_spi_mode(self, mode: int):
        """ Set SPI mode (polarity of clk and data). Possible mode values are 0 and 1. """
        response = self.connection.command(Commands.SET_SPI_MODE(mode))
        if self.debug_spi:
            print(f"SPI mode now {response[0]}")

    def spi_cmd(
            self,
            name: str,
            first: int, second: int, third: int, read: bool,
            fourth: int = 100, show_bin = False, cs = 0, nbyte = 3, quiet = False
    ) -> bytes | None:
        """
        :param name:
        :param first: first byte to send, start of address
        :param second: second byte to send, rest of address
        :param third: third byte to send, value to write, ignored during read
        :param read:
        :param fourth:
        :param show_bin:
        :param cs: cs is which chip to select, adc 0 by default
        :param nbyte: nbyte is 2 or 3, second byte is ignored in case of 2 bytes
        :param quiet:
        """
        if read:
            first = first + 0x80  # set the highest bit for read, i.e. add 0x80

        # Send SPI command result from command
        spi_res = self.connection.command([3, cs, first, second, third, fourth, 100, nbyte])
        if read:
            if not quiet:
                if show_bin:
                    print(f"SPI read:\t{name}(0x{first:X}, 0x{second:X})) => {spi_res[1]:08b}{spi_res[0]:08b}")
                else:
                    print(f"SPI read:\t{name}(0x{first:X}, 0x{second:X})) => 0x{spi_res[1]:X}{spi_res[0]:X}")
            return spi_res
        else:  # write
            if not quiet:
                if nbyte == 4:
                    print(f"SPI write:\t{name}(0x{first:X}, 0x{second:X}) 0x{third:X}, 0x{fourth:X}")
                else:
                    print(f"SPI write:\t{name}(0x{first:X}, 0x{second:X}) 0x{third:X}")

    def spi_cmd2(self, name, first, second, third, fourth, read, cs = 0, nbyte = 3):
        # first byte to send, start of address
        # second byte to send, rest of address
        # third byte to send, value to write, ignored during read, to address +1 (the higher 8 bits)
        # fourth byte to send, value to write, ignored during read
        # cs is which chip to select, adc 0 by default
        # nbyte is 2 or 3, second byte is ignored in case of 2 bytes
        if read: first = first + 0x80  # set the highest bit for read, i.e. add 0x80

        # get SPI result from command
        spires = self.connection.command([3, cs, first, second, fourth, 100, 100, nbyte])

        # get SPI result from command for next byte
        spires2 = self.connection.command([3, cs, first, second + 0x01, third, 100, 100, nbyte])
        if read:
            print(f"SPI read:\t{name}(0x{first:X}, 0x{second:X}) 0x{spires2[0]:X} 0x{spires[0]:X}")
        else:
            print(f"SPI write:\t{name}(0x{first:X}, 0x{second:X}) 0x{fourth:X} 0x{third:X}")

    def adf_reset(self):
        self.adf4350(
            freq = self.sample_rate_GHz * 1000 / 2,
            phase = None,
            themuxout = self.themuxoutV
        )
        time.sleep(0.1)
        res = self.get_boardinbits()
        if bit_asserted(res, 0):
            print(f"Adf pll locked for board {self.connection.board}")
        else:
            print(f"Adf pll for board {self.connection.board} not locked?")  # should be 1 if locked

    def switchclock(self):
        self.clockswitch(True)
        self.clockswitch(False)

    def clockswitch(self, quiet) -> None:
        clockinfo = self.connection.command(Commands.SWITCH_CLOCKS)
        if not quiet:
            print(f"Clock info for board {self.connection.board} {clockinfo[1]:08b} {clockinfo[0]:08b}")
            if bit_asserted(clockinfo[1], 1) and not bit_asserted(clockinfo[1], 3):
                print(f"Board {self.connection.board} locked to ext board")
            else:
                print(f"Board {self.connection.board} locked to internal board")

    def pll_reset(self):
        tres = self.connection.command(Commands.RESET_PLLS)
        print(f"pllreset sent to board {self.connection.board} - got back: {tres}")
        self.phasecs = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # reset counters
        # adjust phases (intentionally put to a place where the clockstr may be bad, it'll get adjusted by 90 deg later, and then dropped to a good range)

        n = 4  # amount to adjust (+ or -)
        for i in range(abs(n)):
            self.dophase(2, n > 0, pllnum = 0, quiet = (i != abs(n) - 1))  # adjust phase of c2, clkout

        n = -1  # amount to adjust (+ or -)
        for i in range(abs(n)):
            self.dophase(3, n > 0, pllnum = 0, quiet = (i != abs(n) - 1))  # adjust phase of c3

        n = 0  # amount to adjust (+ or -)
        for i in range(abs(n)):
            self.dophase(4, n > 0, pllnum = 0, quiet = (i != abs(n) - 1))  # adjust phase of c4

        self.plljustreset = True
        self.switchclock()

    def dophase(self, plloutnum, updown, pllnum = None, quiet = False):
        # for 3rd byte, 000:all 001:M 010=2:C0 011=3:C1 100=4:C2 101=5:C3 110=6:C4
        # for 4th byte, 1 is up, 0 is down
        self.connection.send([6, pllnum, int(plloutnum + 2), updown, 100, 100, 100, 100])
        if updown:
            self.phasecs[pllnum][plloutnum] = self.phasecs[pllnum][plloutnum] + 1
        else:
            self.phasecs[pllnum][plloutnum] = self.phasecs[pllnum][plloutnum] - 1
        if not quiet:
            print(f"phase for pllnum {pllnum} plloutnum {plloutnum} on "
                  f"board {self.connection.board} now {self.phasecs[pllnum][plloutnum]}")

    def get_boardinbits(self) -> int:
        res = self.connection.command(Commands.READ_BOARD_IN)
        print(f"Board in bits {res[0]} :: {res[0]:08b}")
        return res[0]

    def adf4350(self, freq, phase, r_counter = 1, divided = FeedbackSelect.Divider, ref_doubler = False,
                ref_div2 = True,
                themuxout = False):
        print("ADF4350 being set to %0.2f MHz" % freq)
        INT, MOD, FRAC, output_divider, band_select_clock_divider = calculate_regs(
            device_type = DeviceType.ADF4350,
            freq = freq,
            ref_freq = 50.0,
            band_select_clock_mode = BandSelectClockMode.Low,
            feedback_select = divided,
            r_counter = r_counter,  # needed when using FeedbackSelect.Divider (needed for phase resync?!)
            ref_doubler = ref_doubler,
            ref_div2 = ref_div2,
            enable_gcd = True
        )
        print("INT", INT, "MOD", MOD, "FRAC", FRAC, "outdiv", output_divider, "bandselclkdiv",
              band_select_clock_divider)
        regs = make_regs(
            INT = INT, MOD = MOD, FRAC = FRAC, output_divider = output_divider,
            band_select_clock_divider = band_select_clock_divider, r_counter = r_counter, ref_doubler = ref_doubler,
            ref_div_2 = ref_div2,
            device_type = DeviceType.ADF4350, phase_value = phase, mux_out = themuxout, charge_pump_current = 2.50,
            feedback_select = divided, pd_polarity = PDPolarity.Positive, prescaler = '4/5',
            band_select_clock_mode = BandSelectClockMode.Low,
            clk_div_mode = ClkDivMode.ResyncEnable, clock_divider_value = 1000, csr = False,
            aux_output_enable = False, aux_output_power = -4.0, output_enable = True,
            output_power = -4.0
        )  # (-4,-1,2,5)

        # values can also be computed using free Analog Devices ADF435x Software:
        # https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/eval-adf4351.html#eb-relatedsoftware
        self.set_spi_mode(0)
        for r in reversed(range(len(regs))):
            # regs[2]=0x5004E42 #to override from ADF435x software
            print(f"adf4350 reg {r} {regs[r]:08b} 0x{regs[r]:X}")
            fourbytes = int_to_bytes(regs[r])
            self.spi_cmd(
                "ADF4350 Reg " + str(r), fourbytes[3], fourbytes[2], fourbytes[1], False,
                fourth = fourbytes[0],
                cs = 3, nbyte = 4
            )  # was cs=2 on alpha board v1.11

        self.set_spi_mode(0)


def boards() -> list[Board]:
    return [Board(connection) for connection in connect()]

if __name__ == '__main__':
    board = boards()[0]
    print(board)
    waveform: bytes = board.get_waveform()
    print(waveform)
    
    # lvds1bits
    # #1 @0
        # [0:13]
        # [0:11][12][13]
        # data  clk strobe
    # #2 @14
        # [0:13]
        # [0:11][12][13]
        # data  clk strobe
    
    # lvds2bits
    # #11 @0
        # [0:13]
        # [0:11][12][13]
        # data  clk strobe
    ...

    # lvds3bits
    ...

    # lvds4bits
    ...

    # lvdsLbits
    ...


