from hspro_api import InputImpedance, connect, ChannelCoupling, WaveformAvailable, WaveformUnavailable, TriggerType
from hspro_api.mpl_plotter import plot_waveforms

if __name__ == '__main__':
    board = connect(debug=False, show_board_call_trace=True)[0]

    board.enable_two_channels(False)
    board.set_channel_coupling(0, ChannelCoupling.DC)
    board.set_trigger_props(trigger_level=0, trigger_delta=1, trigger_pos=0.75, tot=3, trigger_on_chanel=0)
    board.set_channel_10x_probe(channel=0, ten_x_probe=False)
    board.set_chanel_voltage_div(channel=0, dV=0.2)
    board.set_time_scale("1us")
    board.set_channel_input_impedance(channel=0, impedance=InputImpedance.ONE_MEGA_OHM)

    for i in range(10):
        # board.set_channel_offset_V(channel=0, offset_V=0.15 * i)
        is_armed = board.force_arm_trigger(trigger_type=TriggerType.ON_FALLING_EDGE)
        match board.wait_for_waveform(10):
            case WaveformAvailable(sample_triggered):
                w1, w2 = board.get_waveforms()
                plot_waveforms(w1, w2)

            case WaveformUnavailable():
                print("Waveform unavailable")
                break

    board.cleanup()
