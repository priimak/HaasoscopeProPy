from hspro.waveform import Waveform


def plot_waveform(waveform: Waveform) -> None:
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots()
    ax.grid(True)
    ax.plot(waveform.ts, waveform.vs)
    ax.set_xlabel(waveform.time_unit.to_str())
    ax.set_xlim(waveform.ts[0], waveform.ts[-1])
    dt_per_division = (waveform.ts[-1] - waveform.ts[0]) / 10
    ax.set_ylim(-4 * waveform.dV, 4 * waveform.dV)
    ax.set_xticks([i * dt_per_division + waveform.ts[0] for i in range(11)])
    ax.set_yticks([(i - 4) * waveform.dV for i in range(9)])
    fig.tight_layout()
    plt.show()
