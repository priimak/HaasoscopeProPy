import struct
import time
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from unlib import Duration, TimeUnit

import hspro_api.conn.connection_op
from hspro_api.adf435x_core import FeedbackSelect, calculate_regs, DeviceType, BandSelectClockMode, make_regs, \
    PDPolarity, ClkDivMode
from hspro_api.commands import Commands, TriggerType
from hspro_api.conn.connection import Connection
from hspro_api.registers_enum import RegisterIndex
from hspro_api.time_constants import TimeConstants
from hspro_api.utils import bit_asserted, int_to_bytes, find_longest_zero_stretch
from hspro_api.waveform import Waveform


class WaveformAvailability(ABC):
    pass


class WaveformAvailable(WaveformAvailability):
    __match_args__ = ("sample_triggered",)

    def __init__(self, sample_triggered: int):
        self.sample_triggered = sample_triggered


class WaveformUnavailable(WaveformAvailability):
    pass


class InputImpedance(Enum):
    FIFTY_OHM = 0
    ONE_MEGA_OHM = 1


class ChannelCoupling(Enum):
    AC = 0
    DC = 1


class BoardConsts:
    SAMPLE_RATE_GHZ = 3.2
    NATIVE_SAMPLE_PERIOD_S = 3.125e-10
    VALID_DOWNSAMPLEMERGIN_VALUES_ONE_CHANNEL = [1, 2, 4, 8, 20, 40]
    VALID_DOWNSAMPLEMERGIN_VALUES_TWO_CHANNELS = [1, 2, 4, 10, 20]
    NUM_CHAN_PER_BOARD = 2


@dataclass
class BoardState:
    expect_samples = 100
    restore_samples = 100
    expect_samples_extra = 5  # enough to cover downsample shifting and toff shifting
    samplerate = 3.2  # freq in GHz
    nsunits = 1
    num_logic_inputs = 0
    tenx = 1
    ten_x_probe = [False, False]  # one for each channel
    coupling = [ChannelCoupling.DC, ChannelCoupling.DC]  # channel coupling
    debug = False
    dopattern = 0  # set to 4 to do max varying test pattern
    debugprint = True
    showbinarydata = True
    debugstrobe = False
    dofast = False
    dotwochannel = False
    dointerleaved = False
    dooverrange = False
    total_rx_len = 0
    time_start = time.time()
    triggertype: TriggerType = TriggerType.ON_RISING_EDGE
    isrolling = 0
    selectedchannel = 0
    activeboard = 0
    activexychannel = 0
    tad = 0
    toff = 0
    themuxoutV = True
    phasecs = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    lastclk = -1

    pll_test_c2phase = 5
    pll_test_c2phase_down = 2
    pll_test_c0phase = -5
    pll_test_c0phase_down = 2

    doexttrig = False
    paused = True  # will unpause with dostartstop at startup
    downsample = 0
    downsamplemerging = 1
    highres = True
    downsamplefactor = 1
    xscale = 1
    xscaling = 1

    # this is the size of 1 bit, so that 2^12 bits fill the 10.x divisions on the screen
    yscale = 3.3 / 2.03 * 10 * 5 / 8 / pow(2, 12) / 16
    requested_dV = [1.0, 1.0]
    dV = [1.0, 1.0]
    requested_offset_V = [0.0, 0.0]  # requested voltage offset per channel
    offset_V = [0.0, 0.0]  # actual offset per channel
    configured_trigger_level = [0.0, 0.0]

    min_y = -5  # -pow(2, 11) * yscale
    max_y = 5  # pow(2, 11) * yscale
    min_x = 0
    xydata = 0
    xydatainterleaved = 0
    fftui = 0
    downsamplezoom = 1
    triggerlevel = 127
    triggerdelta = 1
    triggertimethresh = 0
    triggerchan = 0
    hline = 0
    vline = 0
    getone = False
    units = "ns"
    chtext = ""
    linepens = []
    nlines = 0
    statuscounter = 0
    nevents = 0
    oldnevents = 0
    tinterval = 100.
    oldtime = time.time()
    nbadclkA = 0
    nbadclkB = 0
    nbadclkC = 0
    nbadclkD = 0
    nbadstr = 0
    eventcounter = 0
    nsubsamples = 10 * 4 + 8 + 2  # extra 4 for clk+str, and 2 dead beef
    # sample_triggered = [0]
    doeventcounter = False
    fitwidthfraction = 0.2
    extrigboardstdcorrection = 1
    extrigboardmeancorrection = 0
    lastrate = 0
    lastsize = 0
    VperD = [0.16, 0.16]

    plljustreset = -10
    plljustresetdir = 0
    phasenbad = [0] * 12
    phaseoffset = 1  # how many positive phase steps to take from middle of good range

    dooversample = False
    doresamp = 0
    _trigger_pos: float = 0.5
    dt_s: float = BoardConsts.NATIVE_SAMPLE_PERIOD_S
    requested_time_scale: Duration | None = None

    def samples_per_row_per_waveform(self) -> int:
        return 20 if self.dotwochannel else 40

    @property
    def trigger_pos(self) -> float:
        return self._trigger_pos

    @trigger_pos.setter
    def trigger_pos(self, trigger_pos: float) -> None:
        self._trigger_pos = int(self.expect_samples * trigger_pos) / self.expect_samples

    @property
    def trigger_pos_live(self) -> float:
        return self.trigger_pos + 1 / self.expect_samples

    @property
    def absolute_trigger_pos(self) -> int:
        return round(self.expect_samples * self.trigger_pos)

    @property
    def max_x(self) -> float:
        return 4 * 10 * self.expect_samples * self.downsamplefactor / self.nsunits / self.samplerate

    @property
    def configured_trigger_level_V(self) -> list[float]:
        return [self.dV[ch] * 5 * self.configured_trigger_level[ch] for ch in range(2)]


class Board:
    def __init__(
            self,
            connection: Connection,
            debug: bool,
            debug_spi: bool,
            board_trace: Callable[[str], None] = lambda _: None
    ) -> None:
        self.board_trace = board_trace
        self.comm = Commands(connection)
        # self.comm.spi_command("CAL_EN", 0x00, 0x61, 0x00, True)
        self.board_num = connection.board
        self.state = BoardState()

        self.debug = debug
        self.debug_spi = debug_spi

        # configure the board
        # self.cleanup()
        self.reset_adf()
        self.__setupboard()
        for c in range(BoardConsts.NUM_CHAN_PER_BOARD):
            self.__setchanacdc(chan=c, ac=0, doswap=self.state.dooversample)
        self.reset_plls()
        self.comm.auxoutselector(0)

    def set_channel_10x_probe(self, channel: int, ten_x_probe: bool) -> float:
        """
        Set flag on a channel indicating if it is connected to 10x probe. Since this affects voltage/div value,
        new voltage/div value is returned.
        """
        if self.state.ten_x_probe[channel] != ten_x_probe:
            self.state.ten_x_probe[channel] = ten_x_probe
            if ten_x_probe:
                self.state.requested_dV[channel] = self.state.dV[channel] * 10
            else:
                self.state.requested_dV[channel] = self.state.dV[channel] / 10
            self.__update_voltage_div(channel)
            self.set_channel_offset_V(channel, self.state.offset_V[channel])
            retval = self.state.dV[channel]
            self.board_trace(f"set_channel_10x_probe({channel}, {ten_x_probe}) -> {retval}")
            return retval
        else:
            self.board_trace(f"set_channel_10x_probe({channel}, {ten_x_probe}) -> 0")
            return 0.0

    def set_channel_voltage_div(self, channel: int, dV: float) -> float:
        self.state.requested_dV[channel] = dV
        self.__update_voltage_div(channel)
        retval = self.state.dV[channel]
        self.board_trace(f"set_channel_voltage_div({channel}, {dV}) -> {retval}")
        return retval

    def __update_voltage_div(self, channel: int) -> None:
        self.state.dV[channel] = self.comm.set_voltage_div(
            channel=channel,
            dV=self.state.requested_dV[channel],
            do_oversample=False,
            ten_x_probe=self.state.ten_x_probe[channel]
        )

    def enable_two_channels(self, enable: bool) -> None:
        self.board_trace(f"enable_two_channels({enable})")
        if self.state.dotwochannel != enable:
            self.state.dotwochannel = enable
            self.__update_leds()
            self.__setupboard()
            self.__update_time_scale()

    def __update_leds(self):
        if self.state.dotwochannel:
            self.comm.set_led((100, 0, 0), (0, 100, 0))
        else:
            self.comm.set_led((100, 0, 0), (0, 0, 0))

    def set_time_scale(self, time_scale: Duration | str) -> Duration:
        """
        Set time duration per horizontal division. There are 10 divisions. Returns actually set value.
        """
        self.state.requested_time_scale = Duration.value_of(time_scale)
        retval = self.__update_time_scale()
        self.board_trace(f"set_time_scale({time_scale}) -> {retval}")
        return retval

    def get_valid_time_scales(self) -> list[Duration]:
        """
        Returns list of valid durations for horizontal division. This is intended to be used
        in GUI to construct valid time base element.
        """
        # TODO: This function is to be removed
        num_samples_per_division = self.state.samples_per_row_per_waveform() * self.state.expect_samples / 10
        results = []
        for downsample in range(32):
            for downsamplemerging in self.__get_valid_downsamplemergin_values():
                dt_s = BoardConsts.NATIVE_SAMPLE_PERIOD_S * downsamplemerging * pow(2, downsample)
                results.append(Duration.value_of(f"{dt_s * num_samples_per_division} s").optimize())
        return results

    def __update_time_scale(self) -> Duration:
        num_samples_per_division = self.state.samples_per_row_per_waveform() * self.state.expect_samples / 10
        requested_dt = self.state.requested_time_scale / num_samples_per_division
        dt_s, downsample, downsamplemerging = self.__find_downsample_parameters(requested_dt)
        self.state.dt_s = dt_s
        self.state.downsample = downsample
        self.state.downsamplemerging = downsamplemerging
        self.__set_downsample()
        return Duration.value_of(f"{dt_s} s").optimize()

    def __get_valid_downsamplemergin_values(self) -> list[int]:
        if self.state.dotwochannel:
            return BoardConsts.VALID_DOWNSAMPLEMERGIN_VALUES_TWO_CHANNELS
        else:
            return BoardConsts.VALID_DOWNSAMPLEMERGIN_VALUES_ONE_CHANNEL

    def __find_downsample_parameters(self, requested_dt: Duration) -> tuple[float, int, int]:
        for v in (TimeConstants.dt_two_ch if self.state.dotwochannel else TimeConstants.dt_one_ch):
            if v[2] >= requested_dt:
                return v[2].to_float(TimeUnit.S), v[0], v[1]
        raise RuntimeError("Failed to find valid downsample parameters")

    def set_highres_capture_mode(self, highres: bool) -> None:
        self.board_trace(f"set_highres_capture_mode({highres})")
        self.state.highres = highres
        self.__set_downsample()

    def __set_downsample(self):
        self.comm.set_downsample(
            downsample=self.state.downsample,
            highres=self.state.highres,
            downsample_merging=self.state.downsamplemerging
        )

    def force_arm_trigger(self, trigger_type: TriggerType) -> bool:
        retval = self.comm.force_arm_trigger(
            trigger_type=trigger_type,
            two_channels=self.state.dotwochannel,
            oversample=self.state.dooversample,
            absolute_trigger_pos=self.state.absolute_trigger_pos,
            expect_samples=self.state.expect_samples
        )
        self.board_trace(f"force_arm_trigger({trigger_type}) -> {retval}")
        return retval

    def set_trigger_props(
            self,
            trigger_level: float,
            trigger_delta: int,
            trigger_pos: float,
            tot: int,
            trigger_on_channel: int
    ) -> float:
        """
        trigger_level: value in range from -1 to 1; change in step of 0.0078125
        returns configured trigger level
        """
        if trigger_pos < 0 or trigger_pos > 1:
            raise RuntimeError("Trigger position must be between 0 and 1.")
        if trigger_level < -1 or trigger_level > 1:
            raise RuntimeError("Trigger level must be between -1 and 1.")

        t_level = int(min(max(128.0 * trigger_level + 127.0, 0.0), 255))
        if (t_level + trigger_delta >= 256) or (t_level - trigger_delta) <= 0:
            raise RuntimeError("Invalid combindation of trigger level and delta.")

        self.state.trigger_pos = trigger_pos
        self.state.trigger_pos = self.state.absolute_trigger_pos / self.state.expect_samples
        self.comm.set_trigger_props(
            trigger_level=t_level,
            trigger_delta=trigger_delta,
            trigger_pos=self.state.absolute_trigger_pos,
            tot=tot,
            trigger_on_channel=trigger_on_channel
        )
        self.comm.set_prelength_to_take(self.state.absolute_trigger_pos + 4)
        self.state.configured_trigger_level[trigger_on_channel] = trigger_level
        configured_level = (t_level - 127.0) / 128.0
        self.board_trace(
            f"set_trigger_props({trigger_level}, {trigger_delta}, {trigger_pos}, "
            f"{tot}, {trigger_on_channel}) -> {configured_level}"
        )
        return configured_level

    def wait_for_calibration_done(self):
        # Poll for ADC calibration to be complete. This is reflected in bit 7 or `boardin` register.
        # Raise error if calibration is not done within 1 second.
        self.board_trace("wait_for_calibration_done()")
        start_at = time.time()
        while True:
            boardin = self.comm.read_register(RegisterIndex.boardin)
            if (boardin & 0b10000000) > 0:
                break
            elif (time.time() - start_at > 3):
                # self.cleanup()
                self.comm.get_boardin()
                raise RuntimeError("ADC calibration on the board did not complete within 3 seconds.")

    def reset_adf(self):
        self.board_trace("reset_adf()")
        self.__adf4350(
            freq=BoardConsts.SAMPLE_RATE_GHZ * 1000 / 2,
            phase=None,
            themuxout=self.state.themuxoutV
        )
        time.sleep(0.1)
        res = self.comm.get_boardin()
        if self.debug:
            if bit_asserted(res, 5):
                print(f"Adf pll locked for board {self.board_num}")
            else:
                print(f"Adf pll for board {self.board_num} not locked?")  # should be 1 if locked

    def __adf4350(
            self, freq: float, phase, r_counter=1, divided=FeedbackSelect.Divider, ref_doubler=False,
            ref_div2=True,
            themuxout=False
    ):
        if self.debug:
            print('ADF4350 being set to %0.2f MHz' % freq)
        INT, MOD, FRAC, output_divider, band_select_clock_divider = calculate_regs(
            device_type=DeviceType.ADF4350, freq=freq, ref_freq=50.0,
            band_select_clock_mode=BandSelectClockMode.Low,
            feedback_select=divided,
            r_counter=r_counter,  # needed when using FeedbackSelect.Divider (needed for phase resync?!)
            ref_doubler=ref_doubler, ref_div2=ref_div2, enable_gcd=True)

        if self.debug:
            print(f"INT: {INT}, MOD: {MOD}, FRAC: {FRAC}, outdiv: {output_divider}, "
                  f"bandselclkdiv: {band_select_clock_divider}")

        regs = make_regs(
            INT=INT, MOD=MOD, FRAC=FRAC, output_divider=output_divider,
            band_select_clock_divider=band_select_clock_divider, r_counter=r_counter, ref_doubler=ref_doubler,
            ref_div_2=ref_div2,
            device_type=DeviceType.ADF4350, phase_value=phase, mux_out=themuxout, charge_pump_current=2.50,
            feedback_select=divided, pd_polarity=PDPolarity.Positive, prescaler='4/5',
            band_select_clock_mode=BandSelectClockMode.Low,
            clk_div_mode=ClkDivMode.ResyncEnable, clock_divider_value=1000, csr=False,
            aux_output_enable=False, aux_output_power=-4.0, output_enable=True, output_power=-4.0
        )  # (-4,-1,2,5)
        # values can also be computed using free Analog Devices ADF435x Software:
        # https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/eval-adf4351.html#eb-relatedsoftware
        self.comm.set_spi_mode(0)

        for r in reversed(range(len(regs))):
            # regs[2]=0x5004E42 #to override from ADF435x software
            if self.debug:
                print(f"adf4350 reg {r} = {regs[r]:08b} 0x{regs[r]:02X}")

            fourbytes = int_to_bytes(regs[r])
            # for i in range(4): print(binprint(fourbytes[i]))
            self.comm.spi_command(
                "ADF4350 Reg " + str(r), fourbytes[3], fourbytes[2], fourbytes[1], False,
                fourth=fourbytes[0],
                cs=3, nbyte=4  # was cs=2 on alpha board v1.11
            )
        self.comm.set_spi_mode(0)

    def reset_plls(self):
        self.board_trace("reset_plls()")
        self.comm.reset_plls()
        self.state.phasecs = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # reset counters

        self.state.plljustreset = 0
        self.state.plljustresetdir = 1
        self.state.phasenbad = [0] * 12  # reset nbad counters
        self.state.restore_samples = self.state.expect_samples
        self.state.expect_samples = 1000
        self.__switchclock()

        # adjust phases (intentionally put to a place where the clockstr may be bad, it'll get adjusted
        # by 90 deg later, and then dropped to a good range)
        # n = self.state.pll_test_c0phase  # amount to adjust (+ or -)
        # for i in range(abs(n)):
        #     self.__dophase(0, n > 0, pllnum=0, quiet=(i != abs(n) - 1))  # adjust phase of c0, clklvds
        # self.state.plljustreset = 3  # get a few events

    def __dophase(self, plloutnum: int, updown: bool, pllnum: int, quiet=False):
        # for 3rd byte, 000:all 001:M 010=2:C0 011=3:C1 100=4:C2 101=5:C3 110=6:C4
        # for 4th byte, 1 is up, 0 is down
        self.comm.set_clk_phase_adjust(pll_num=pllnum, pll_out_num=plloutnum, up_down=updown)
        self.state.phasecs[pllnum][plloutnum] = self.state.phasecs[pllnum][plloutnum] + (1 if updown else -1)

        if not quiet and self.debug:
            print(
                "phase for pllnum", pllnum,
                "plloutnum", plloutnum,
                "on board", self.board_num,
                "now", self.state.phasecs[pllnum][plloutnum]
            )

    def __switchclock(self):
        self.clockswitch(True)
        self.clockswitch(False)

    def clockswitch(self, quiet: bool):
        self.board_trace("clockswitch()")
        clockinfo = self.comm.clk_switch()
        if self.debug and not quiet:
            print(f"Clockinfo for board {self.board_num} {clockinfo:08b}")
            if bit_asserted(clockinfo, 1) and not bit_asserted(clockinfo, 3):
                print(f"Board {self.board_num} locked to ext board")
            else:
                print(f"Board {self.board_num} locked to internal clock")

    def __setupboard(self):
        self.comm.set_fanon(True)

        self.comm.set_spi_mode(0)
        self.comm.spi_command("DEVICE_CONFIG", 0x00, 0x02, 0x00, False)  # power up
        # self.comm.spi_command("DEVICE_CONFIG", 0x00, 0x02, 0x03, False) # power down
        self.comm.spi_command2("VENDOR", 0x00, 0x0c, 0x00, 0x00, True)
        self.comm.spi_command("LVDS_EN", 0x02, 0x00, 0x00, False)  # disable LVDS interface
        self.comm.spi_command("CAL_EN", 0x00, 0x61, 0x00, False)  # disable calibration

        if self.state.dotwochannel:
            self.comm.spi_command("LMODE", 0x02, 0x01, 0x03, False)  # LVDS mode: aligned, demux, dual channel, 12-bit
            # spicommand("LMODE", 0x02, 0x01, 0x01, False)  # LVDS mode: staggered, demux, dual channel, 12-bit
        else:
            self.comm.spi_command("LMODE", 0x02, 0x01, 0x07, False)  # LVDS mode: aligned, demux, single channel, 12-bit
            # self.comm.spi_command("LMODE", 0x02, 0x01, 0x37, False)  # LVDS mode: aligned, demux, single channel, 8-bit
            # self.comm.spi_command("LMODE", 0x02, 0x01, 0x05, False)  # LVDS mode: staggered, demux, single channel, 12-bit

        self.comm.spi_command("LVDS_SWING", 0x00, 0x48, 0x00, False)  # high swing mode
        # self.comm.spi_command("LVDS_SWING", 0x00, 0x48, 0x01, False)  #low swing mode

        self.comm.spi_command("LCTRL", 0x02, 0x04, 0x0a, False)  # use LSYNC_N (software), 2's complement
        # self.comm.spi_command("LCTRL", 0x02, 0x04, 0x08, False)  # use LSYNC_N (software), offset binary

        self.__swapinputs(doswap=False, insetup=True)

        # self.comm.spi_command("TAD", 0x02, 0xB7, 0x01, False)  # invert clk
        self.comm.spi_command("TAD", 0x02, 0xB7, 0x00, False)  # don't invert clk

        tad = 0
        self.comm.spi_command("TAD", 0x02, 0xB6, tad, False)  # adjust TAD (time of ADC relative to clk)

        if self.state.dooverrange:
            self.comm.spi_command("OVR_CFG", 0x02, 0x13, 0x0f, False)  # overrange on
            self.comm.spi_command("OVR_T0", 0x02, 0x11, 0xf2, False)  # overrange threshold 0
            self.comm.spi_command("OVR_T1", 0x02, 0x12, 0xab, False)  # overrange threshold 1
        else:
            self.comm.spi_command("OVR_CFG", 0x02, 0x13, 0x07, False)  # overrange off

        if self.state.dopattern != 0:
            self.comm.spi_command("PAT_SEL", 0x02, 0x05, 0x11, False)  # test pattern
            usrval = 0x00
            if self.state.dopattern == 1:
                self.comm.spi_command2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.comm.spi_command2("UPAT1", 0x01, 0x82, usrval, usrval + 1, False)  # set pattern sample 1
                self.comm.spi_command2("UPAT2", 0x01, 0x84, usrval, usrval + 2, False)  # set pattern sample 2
                self.comm.spi_command2("UPAT3", 0x01, 0x86, usrval, usrval + 4, False)  # set pattern sample 3
                self.comm.spi_command2("UPAT4", 0x01, 0x88, usrval, usrval + 8, False)  # set pattern sample 4
                self.comm.spi_command2("UPAT5", 0x01, 0x8a, usrval, usrval + 16, False)  # set pattern sample 5
                self.comm.spi_command2("UPAT6", 0x01, 0x8c, usrval, usrval + 32, False)  # set pattern sample 6
                self.comm.spi_command2("UPAT7", 0x01, 0x8e, usrval, usrval + 64, False)  # set pattern sample 7
            elif self.state.dopattern == 2:
                self.comm.spi_command2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.comm.spi_command2("UPAT1", 0x01, 0x82, usrval + 1, usrval, False)  # set pattern sample 1
                self.comm.spi_command2("UPAT2", 0x01, 0x84, usrval, usrval, False)  # set pattern sample 2
                self.comm.spi_command2("UPAT3", 0x01, 0x86, usrval + 2, usrval, False)  # set pattern sample 3
                self.comm.spi_command2("UPAT4", 0x01, 0x88, usrval, usrval, False)  # set pattern sample 4
                self.comm.spi_command2("UPAT5", 0x01, 0x8a, usrval + 3, usrval, False)  # set pattern sample 5
                self.comm.spi_command2("UPAT6", 0x01, 0x8c, usrval, usrval, False)  # set pattern sample 6
                self.comm.spi_command2("UPAT7", 0x01, 0x8e, usrval + 4, usrval, False)  # set pattern sample 7
            elif self.state.dopattern == 3:
                self.comm.spi_command2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.comm.spi_command2("UPAT1", 0x01, 0x82, usrval + 0x01, usrval + 0x01, False)  # set pattern sample 1
                self.comm.spi_command2("UPAT2", 0x01, 0x84, usrval + 0x01, usrval + 0x03, False)  # set pattern sample 2
                self.comm.spi_command2("UPAT3", 0x01, 0x86, usrval + 0x03, usrval + 0x07, False)  # set pattern sample 3
                self.comm.spi_command2("UPAT4", 0x01, 0x88, usrval + 0x03, usrval + 0x0f, False)  # set pattern sample 4
                self.comm.spi_command2("UPAT5", 0x01, 0x8a, usrval + 0x07, usrval + 0x7f, False)  # set pattern sample 5
                self.comm.spi_command2("UPAT6", 0x01, 0x8c, usrval + 0x07, usrval + 0xff, False)  # set pattern sample 6
                self.comm.spi_command2("UPAT7", 0x01, 0x8e, usrval + 0x08, usrval, False)  # set pattern sample 7
            elif self.state.dopattern == 4:
                self.comm.spi_command2("UPAT0", 0x01, 0x80, usrval, usrval, False)  # set pattern sample 0
                self.comm.spi_command2("UPAT1", 0x01, 0x82, usrval + 0x0f, usrval + 0xff, False)  # set pattern sample 1
                self.comm.spi_command2("UPAT2", 0x01, 0x84, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 2
                self.comm.spi_command2("UPAT3", 0x01, 0x86, usrval + 0x0f, usrval + 0xff, False)  # set pattern sample 3
                self.comm.spi_command2("UPAT4", 0x01, 0x88, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 4
                self.comm.spi_command2("UPAT5", 0x01, 0x8a, usrval + 0x0f, usrval + 0xff, False)  # set pattern sample 5
                self.comm.spi_command2("UPAT6", 0x01, 0x8c, usrval + 0x00, usrval + 0x00, False)  # set pattern sample 6
                self.comm.spi_command2("UPAT7", 0x01, 0x8e, usrval + 0x0f, usrval + 0xff, False)  # set pattern sample 7
            # self.comm.spi_command("UPAT_CTRL", 0x01, 0x90, 0x0e, False)  # set lane pattern to user, invert a bit of B C D
            self.comm.spi_command("UPAT_CTRL", 0x01, 0x90, 0x00, False)  # set lane pattern to user
        else:
            self.comm.spi_command("PAT_SEL", 0x02, 0x05, 0x02, False)  # normal ADC data
            self.comm.spi_command("UPAT_CTRL", 0x01, 0x90, 0x1e, False)  # set lane pattern to default

        self.comm.spi_command("CAL_EN", 0x00, 0x61, 0x01, False)  # enable calibration
        # self.wait_for_calibration_done()

        self.comm.spi_command("LVDS_EN", 0x02, 0x00, 0x01, False)  # enable LVDS interface
        self.comm.spi_command("LSYNC_N", 0x02, 0x03, 0x00, False)  # assert ~sync signal
        self.comm.spi_command("LSYNC_N", 0x02, 0x03, 0x01, False)  # deassert ~sync signal
        # self.comm.spi_command("CAL_SOFT_TRIG", 0x00, 0x6c, 0x00, False)
        # self.comm.spi_command("CAL_SOFT_TRIG", 0x00, 0x6c, 0x01, False)

        self.comm.set_spi_mode(0)
        self.comm.spi_command("Amp Rev ID", 0x00, 0x00, 0x00, True, cs=1, nbyte=2)
        self.comm.spi_command("Amp Prod ID", 0x01, 0x00, 0x00, True, cs=1, nbyte=2)
        self.comm.spi_command("Amp Rev ID", 0x00, 0x00, 0x00, True, cs=2, nbyte=2)
        self.comm.spi_command("Amp Prod ID", 0x01, 0x00, 0x00, True, cs=2, nbyte=2)

        self.comm.set_spi_mode(1)
        self.comm.spi_command("DAC ref on", 0x38, 0xff, 0xff, False, cs=4)
        self.comm.spi_command("DAC gain 1", 0x02, 0xff, 0xff, False, cs=4)
        self.comm.set_spi_mode(0)
        self.__dooffset(chan=0, val=0, scaling=1, doswap=False)
        self.__dooffset(chan=1, val=0, scaling=1, doswap=False)
        self.__setgain(chan=0, value=0, doswap=False)
        self.__setgain(chan=1, value=0, doswap=False)

    def __swapinputs(self, doswap: bool, insetup: bool):
        if not insetup:
            self.comm.spi_command("LVDS_EN", 0x02, 0x00, 0x00, False)  # disable LVDS interface
            self.comm.spi_command("CAL_EN", 0x00, 0x61, 0x00, False)  # disable calibration

        if doswap:
            self.comm.spi_command("INPUT_MUX", 0x00, 0x60, 0x12, False)  # swap inputs
        else:
            self.comm.spi_command("INPUT_MUX", 0x00, 0x60, 0x01, False)  # unswap inputs

        if not insetup:
            self.comm.spi_command("CAL_EN", 0x00, 0x61, 0x01, False)  # enable calibration
            self.comm.spi_command("LVDS_EN", 0x02, 0x00, 0x01, False)  # enable LVDS interface

    def set_channel_offset_V(self, channel: int, offset_V: float) -> float:
        saved_offset_value = self.state.requested_offset_V[channel]

        self.state.requested_offset_V[channel] = offset_V
        new_offset_value: float | None = self.__set_channel_offset(channel)
        if new_offset_value is None:
            self.state.requested_offset_V[channel] = saved_offset_value
        else:
            self.state.offset_V[channel] = new_offset_value

        self.board_trace(f"set_channel_offset_V({channel}, {offset_V}) -> {self.state.offset_V[channel]}")
        return self.state.offset_V[channel]

    def __set_channel_offset(self, channel: int) -> float | None:
        scaling = 1000 * self.state.dV[channel] / 160

        # offset gain is different in AC mode
        scaling = scaling * (1 if self.state.coupling[channel] == ChannelCoupling.DC else 245 / 160)

        n = int(self.state.requested_offset_V[channel] * 1000 / (1.5 * scaling))
        actual_offset_V = 1.5 * scaling * n / 1000

        # offset gain is different in AC mode
        actual_offset_V = actual_offset_V * (1 if self.state.coupling[channel] == ChannelCoupling.DC else 160 / 245)

        scl = scaling / 10 if self.state.ten_x_probe[channel] else scaling
        if self.__dooffset(chan=channel, val=n, scaling=scl, doswap=self.state.dooversample):
            return actual_offset_V
        else:
            return None

    def __dooffset(self, chan: int, val, scaling, doswap) -> bool:
        # if doswap: val= -val
        dacval = int((pow(2, 16) - 1) * (val * scaling / 2 + 500) / 1000)
        # print("dacval is", dacval,"and doswap is",doswap,"and val is",val)
        if 0 < dacval < pow(2, 16):
            self.comm.set_spi_mode(1)
            if doswap: chan = (chan + 1) % 2
            if chan == 1: self.comm.spi_command("DAC 1 value", 0x18, dacval >> 8, dacval % 256, False, cs=4, quiet=True)
            if chan == 0: self.comm.spi_command("DAC 2 value", 0x19, dacval >> 8, dacval % 256, False, cs=4, quiet=True)
            self.comm.set_spi_mode(0)
            return True
        else:
            return False

    def __setgain(self, *, chan: int, value: int, doswap: bool):
        self.comm.set_spi_mode(0)
        # 00 to 20 is 26 to -6 dB, 0x1a is no gain
        if doswap: chan = (chan + 1) % 2
        if chan == 0: self.comm.spi_command("Amp Gain 0", 0x02, 0x00, 26 - value, False, cs=2, nbyte=2, quiet=True)
        if chan == 1: self.comm.spi_command("Amp Gain 1", 0x02, 0x00, 26 - value, False, cs=1, nbyte=2, quiet=True)

    def __setchanacdc(self, chan: int, ac: int, doswap: bool):
        if doswap: chan = (chan + 1) % 2
        match chan:
            case 0:
                self.comm.set_boardout(control_bit=1, value_bit=(0 if ac == 1 else 1))
            case 1:
                self.comm.set_boardout(control_bit=5, value_bit=(0 if ac == 1 else 1))
            case _:
                raise RuntimeError(f"Invalid channel number {chan}")

    def set_channel_coupling(self, channel: int, channel_coupling: ChannelCoupling) -> None:
        self.board_trace(f"set_channel_coupling({channel}, {channel_coupling})")
        chan = ((channel + 1) % 2) if self.state.dooversample else channel
        match chan:
            case 0:
                self.comm.set_boardout(control_bit=1, value_bit=channel_coupling.value)
            case 1:
                self.comm.set_boardout(control_bit=5, value_bit=channel_coupling.value)
            case _:
                raise RuntimeError(f"Invalid channel number {chan}")
        self.state.coupling[channel] = channel_coupling

        # changing coupling changes offset scaling factors and thus require us to set channel offset again
        self.__set_channel_offset(channel)

    def set_channel_5x_attenuation(self, channel, att: bool):
        self.board_trace(f"set_channel_5x_attenuation({channel}, {att})")
        chan = ((channel + 1) % 2) if self.state.dooversample else channel
        match chan:
            case 0:
                self.comm.set_boardout(control_bit=2, value_bit=(1 if att else 0))
            case 1:
                self.comm.set_boardout(control_bit=6, value_bit=(1 if att else 0))
            case _:
                raise RuntimeError(f"Invalid channel number {chan}")

    def cleanup(self):
        self.board_trace("cleanup()")
        self.comm.set_spi_mode(0)
        self.comm.spi_command("DEVICE_CONFIG", 0x00, 0x02, 0x03, False)  # power down
        self.comm.set_fanon(False)
        self.comm.set_led((0x0f, 0x0f, 0x0f), (0x0f, 0x0f, 0x0f))
        self.comm.set_led((0x0f, 0x0f, 0x0f), (0x0f, 0x0f, 0x0f))
        self.comm.set_configured(False)

    def __downsample_merging_counter_triggered(self) -> tuple[int, int]:
        if self.state.downsamplemerging == 0:
            return 0
        else:
            downsamplemergingcounter, triggerphase = self.comm.get_downsample_merging_counter_triggered()
            if self.state.downsamplemerging <= 1:
                downsamplemergingcounter = 0

            if downsamplemergingcounter == self.state.downsamplemerging and not self.state.doexttrig:
                return 0, triggerphase
            else:
                return downsamplemergingcounter, triggerphase

    def __get_sample_triggered(self) -> int:
        b = self.comm.read_register(RegisterIndex.sample_triggered).to_bytes(4)
        sample_triggered = 19 - f"{b[1]:04b}{b[2]:08b}{b[3]:08b}".find("10")
        if sample_triggered > 19:
            return 0
        else:
            return sample_triggered

    def get_waveforms(self) -> tuple[Waveform, Waveform | None]:
        sample_triggered = self.__get_sample_triggered()

        if self.state.doeventcounter:
            new_event_counter = self.comm.get_eventconter()
            if new_event_counter != self.state.eventcounter + 1 and new_event_counter != 0:
                # check event count, but account for rollover
                print("Event counter not incremented by 1?", new_event_counter, self.state.eventcounter,
                      " for board", self.board_num)
            self.state.eventcounter = new_event_counter

        downsample_merging_counter_triggered, trigger_phase = self.__downsample_merging_counter_triggered()

        # length to request: each adc bit is stored as 10 bits in 2 bytes, a couple extra for shifting later
        expect_len = (self.state.expect_samples + self.state.expect_samples_extra) * 2 * self.state.nsubsamples
        waveform_data = self.comm.get_waveform_data(expect_len)

        # file = (Path.home() / "tmp" / "waveform.bin")
        # file.write_bytes(waveform_data)

        rx_len = len(waveform_data)
        self.state.total_rx_len += rx_len
        trace_1, trace_2 = self.__parse_waveform_data(
            waveform_data, self.state.expect_samples, downsample_merging_counter_triggered,
            sample_triggered, trigger_phase
        )
        pos = self.state.absolute_trigger_pos * self.state.samples_per_row_per_waveform()

        if self.state.dotwochannel:
            return (
                Waveform(self.state.dt_s, trace_1, pos, self.state.dV[0], self.state.configured_trigger_level_V[0]),
                Waveform(self.state.dt_s, trace_2, pos, self.state.dV[1], self.state.configured_trigger_level_V[1])
            )
        else:
            return (
                Waveform(self.state.dt_s, trace_1, pos, self.state.dV[0], self.state.configured_trigger_level_V[0]),
                None
            )

    def __parse_waveform_data(
            self, data: bytes, expect_samples: int, downsample_merging_counter: int, sample_triggered: int,
            trigger_phase: int
    ) -> tuple[list[float], list[float]]:
        nbadclkA = 0
        nbadclkB = 0
        nbadclkC = 0
        nbadclkD = 0
        nbadstr = 0

        one_channel_trace_len = 40 * expect_samples
        two_channel_trace_len = 20 * expect_samples
        trace_A = [0] * (two_channel_trace_len if self.state.dotwochannel else one_channel_trace_len)
        trace_B = [0] * (two_channel_trace_len if self.state.dotwochannel else 0)
        traces: tuple[list[float], list[float]] = trace_A, trace_B

        unpackedsamples = struct.unpack('<' + 'h' * (len(data) // 2), data)
        npunpackedsamples = np.array(unpackedsamples, dtype='float')
        npunpackedsamples *= self.state.yscale

        downsampleoffset = 2 * (
                sample_triggered + (downsample_merging_counter - 1) % self.state.downsamplemerging * 10
        ) // self.state.downsamplemerging
        if not self.state.dotwochannel: downsampleoffset *= 2
        # if self.doexttrig[board]: downsampleoffset -= self.toff // self.downsamplefactor
        # datasize = self.xydata[board][1].size
        datasize = 10 * self.state.expect_samples * (2 if self.state.dotwochannel else 4)

        for s in range(0, expect_samples + self.state.expect_samples_extra):
            nsubsamples = self.state.nsubsamples
            vals = unpackedsamples[s * nsubsamples + 40:s * nsubsamples + 50]
            if vals[9] != -16657: print("no beef?")  # -16657 is 0xbeef
            if vals[8] != 0 or (self.state.lastclk != 341 and self.state.lastclk != 682):
                # only bother checking if there was a clkstr problem detected in firmware, or we need to decode 
                # because of a previous clkstr prob and now want to update self.lastclk
                for n in range(0, 8):  # the subsample to get
                    val = vals[n]
                    if n < 4:
                        if val != 341 and val != 682:  # 0101010101 or 1010101010
                            if n == 0: nbadclkA += 1
                            if n == 1: nbadclkB += 1
                            if n == 2: nbadclkC += 1
                            if n == 3: nbadclkD += 1
                            # print("s=", s, "n=", n, "clk", val, binprint(val))
                        self.state.lastclk = val
                    elif val not in {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512}:
                        nbadstr = nbadstr + 1
                        # print("s=", s, "n=", n, "str", val, binprint(val))
            if self.state.dotwochannel:
                samp = s * 20 - downsampleoffset - (trigger_phase >> 4) // 2
                nsamp = 20
                nstart = 0
                if samp < 0:
                    nsamp = 20 + samp
                    nstart = -samp
                    samp = 0
                if samp + 20 >= datasize:
                    nsamp = datasize - samp
                if 0 < nsamp <= 20:
                    # ch = 1 if n < 20 else 0
                    traces[0][samp:samp + nsamp] = npunpackedsamples[
                                                   s * nsubsamples + 20 + nstart:s * nsubsamples + 20 + nstart + nsamp]

                    traces[1][samp:samp + nsamp] = npunpackedsamples[
                                                   s * nsubsamples + nstart:s * nsubsamples + nstart + nsamp]
            else:
                samp = s * 40 - downsampleoffset - (trigger_phase >> 4)
                nsamp = 40
                nstart = 0
                if samp < 0:
                    nsamp = 40 + samp
                    nstart = -samp
                    samp = 0
                if samp + 40 >= datasize:
                    nsamp = datasize - samp
                if 0 < nsamp <= 40:
                    traces[0][samp:samp + nsamp] = npunpackedsamples[
                                                   s * nsubsamples + nstart:s * nsubsamples + nstart + nsamp]

        self.__adjustclocks(nbadclkA, nbadclkB, nbadclkC, nbadclkD, nbadstr)

        self.__nbadclkA = nbadclkA
        self.__nbadclkB = nbadclkB
        self.__nbadclkC = nbadclkC
        self.__nbadclkD = nbadclkD
        self.__nbadstr = nbadstr

        return traces

    def __adjustclocks(self, nbadclkA, nbadclkB, nbadclkC, nbadclkD, nbadstr):
        plloutnum = 0  # adjusting clklvds
        plloutnum2 = 1  # adjusting clklvdsout
        if 0 <= self.state.plljustreset < 12:  # we start by going up in phase
            nbad = nbadclkA + nbadclkB + nbadclkC + nbadclkD + nbadstr
            self.state.phasenbad[self.state.plljustreset] += nbad
            # adjust phase of plloutnum and plloutnum2
            self.__dophase(plloutnum, (self.state.plljustresetdir == 1), pllnum=0, quiet=True)
            self.__dophase(plloutnum2, (self.state.plljustresetdir == 1), pllnum=0, quiet=True)

            self.state.plljustreset += self.state.plljustresetdir

        elif self.state.plljustreset >= 12:
            if self.state.plljustreset == 15: self.state.plljustresetdir = -1
            self.state.plljustreset += self.state.plljustresetdir
            # adjust phase of plloutnum and plloutnum2
            self.__dophase(plloutnum, (self.state.plljustresetdir == 1), pllnum=0, quiet=True)
            self.__dophase(plloutnum2, (self.state.plljustresetdir == 1), pllnum=0, quiet=True)

        elif self.state.plljustreset == -1:
            print("bad clkstr per phase step:", self.state.phasenbad)
            startofzeros, lengthofzeros = find_longest_zero_stretch(self.state.phasenbad, True)
            print("good phase starts at", startofzeros, "and goes for", lengthofzeros, "steps")
            if startofzeros >= 12: startofzeros -= 12
            n = startofzeros + lengthofzeros // 2 + self.state.phaseoffset  # amount to adjust clklvds (positive)
            if n >= 12: n -= 12
            n += 1  # extra 1 because we went to phase=-1 before
            for i in range(n):
                self.__dophase(plloutnum, True, pllnum=0, quiet=(i != n - 1))  # adjust phase of plloutnum
                self.__dophase(plloutnum2, True, pllnum=0, quiet=(i != n - 1))  # adjust phase of plloutnum
            self.state.plljustreset += self.state.plljustresetdir

        elif self.state.plljustreset == -2:
            self.set_memory_depth(self.state.restore_samples)
            self.state.plljustreset += self.state.plljustresetdir

        elif self.state.plljustreset == -3:
            self.state.plljustreset += self.state.plljustresetdir

    def force_data_acquisition(self) -> bool:
        n_tries = 20
        for _ in range(n_tries):
            result = self.comm.force_data_acquisition()
            if result == 2:
                return True
            elif result == 0:
                # dorolling is enabled, and we cannot force data acquisition unless dorolling is disabled
                return False

        raise RuntimeError(f"Failed to force data acquisition after {n_tries} tries.")

    def wait_for_waveform(self, timeout_s: float = -1) -> WaveformAvailability:
        if timeout_s < 0:
            while True:
                is_available, sample_triggered = self.comm.is_capture_available()
                if is_available:
                    return WaveformAvailable(sample_triggered)
                else:
                    time.sleep(0.0125)  # 1/80 of a second

        elif timeout_s == 0:
            # a single try
            is_available, sample_triggered = self.comm.is_capture_available()
            if is_available:
                return WaveformAvailable(sample_triggered)
            else:
                return WaveformUnavailable()

        else:  # timeout_s > 0
            start_at_s = time.time()
            while time.time() <= start_at_s + timeout_s:
                is_available, sample_triggered = self.comm.is_capture_available()
                if is_available:
                    return WaveformAvailable(sample_triggered)
                else:
                    time.sleep(0.0125)  # 1/80 of a second

            # if we are here timeout expired and waveform is still not available
            return WaveformUnavailable()

    def set_memory_depth(self, depth: int) -> None:
        """
        Set total number of data point blocks to capture. Each block is 40 points in single channel
        mode and 20 in double channel mode.
        """
        self.board_trace(f"set_memory_depth({depth})")
        self.state.expect_samples = depth
        self.state.restore_samples = depth

    def set_channel_input_impedance(self, channel: int, impedance: InputImpedance) -> None:
        self.board_trace(f"set_channel_input_impedance({channel}, {impedance})")
        chan = ((channel + 1) % 2) if self.state.dooversample else channel
        match chan:
            case 0:
                self.comm.set_boardout(0, impedance.value)
            case 1:
                self.comm.set_boardout(4, impedance.value)
            case _:
                raise RuntimeError("Channel must be 0 or 1.")

    def num_samples_per_division(self):
        return self.state.samples_per_row_per_waveform() * self.state.expect_samples / 10


def mk_board(connection: Connection, debug: bool, debug_spi: bool, show_board_call_trace: bool = False):
    tracer = lambda _: None
    if show_board_call_trace:
        tracer = lambda s: print(s)
    return Board(connection, debug, debug_spi, tracer)


def connect(debug: bool = False, debug_spi: bool = False, show_board_call_trace: bool = False) -> list[Board]:
    boards = []
    for connection in hspro_api.conn.connection_op.connect(debug):
        try:
            boards.append(mk_board(connection, debug, debug_spi, show_board_call_trace))
        except Exception as ex:
            print(f"Failed to setup board: {ex}")

    return boards
