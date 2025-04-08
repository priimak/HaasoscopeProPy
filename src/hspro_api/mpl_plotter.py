from hspro_api.waveform import Waveform


def plot_waveforms(waveform_1: Waveform, waveform_2: Waveform | None) -> None:
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots()
    ax.grid(True)
    ax.plot(waveform_1.ts, waveform_1.vs)
    if waveform_2 is not None:
        ax.plot(waveform_2.ts, waveform_2.vs)
    ax.set_xlabel(waveform_1.time_unit.to_str())
    ax.set_xlim(waveform_1.ts[0], waveform_1.ts[-1])
    dt_per_division = (waveform_1.ts[-1] - waveform_1.ts[0]) / 10
    ax.set_ylim(-5 * waveform_1.dV, 5 * waveform_1.dV)
    ax.set_xticks([i * dt_per_division + waveform_1.ts[0] for i in range(11)])
    ax.set_yticks([(i - 4) * waveform_1.dV for i in range(11)])
    ax.axvline(0, linewidth=2, color="red")
    if waveform_1.trigger_level_V is not None:
        ax.axhline(waveform_1.trigger_level_V, color="red")
    fig.tight_layout()
    plt.show()
