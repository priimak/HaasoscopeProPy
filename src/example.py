import time

from matplotlib import pyplot as plt

from hspro.board import connect, WaveformAvailable, WaveformUnavailable
from hspro.commands import TriggerType
from hspro.mpl_plotter import plot_waveform
from hspro.registers_enum import RegisterIndex

if __name__ == '__main__':
    board = connect(debug=True)[0]

    board.set_trigger_props(trigger_level=130, trigger_delta=1, trigger_pos=0.75, tot=1, trigger_on_chanel=0)
    board.set_voltage_div(channel=0, dV=0.1, ten_x_probe=False)
    board.comm.set_offset_V(channel=0, offsetV=0, do_oversample=False, ten_x_probe=False)
    board.set_time_scale("10ns")

    for i in range(10):
        is_armed = board.force_arm_trigger(trigger_type=TriggerType.AUTO)
        match board.wait_for_waveform(10):
            case WaveformAvailable(sample_triggered):
                waveform = board.get_waveform()
                plot_waveform(waveform)

            case WaveformUnavailable(): print("Waveform unavailable")

    board.cleanup()
