from functools import cache

from unlib import TimeUnit, Duration


class Waveform:
    def __init__(self, dt_s: float, vs: list[float], trigger_pos: int, dV: float, trigger_level_V: float | None):
        self.dt_s = dt_s

        time_window_s = dt_s * len(vs)
        time_per_division_s = time_window_s / 10
        self.time_unit: TimeUnit = Duration.value_of(f"{time_per_division_s} s").optimize().time_unit
        dt = Duration.value_of(f"{dt_s}s").in_unit(self.time_unit)

        self.ts: list[float] = [(dt * (i - trigger_pos)).value for i in range(len(vs))]
        self.vs = vs
        self.trigger_pos = trigger_pos
        self.dV = dV
        self.trigger_level_V = trigger_level_V

    def apply_trigger_correction(self, trigger_pos: float):
        self.trigger_pos = int(len(self.vs) * trigger_pos)

    @cache
    def get_t_vec(self, time_unit: TimeUnit) -> list[float]:
        dt = Duration.value_of(f"{self.dt_s} s").to_float(time_unit)
        return [(dt * (i - self.trigger_pos)) for i in range(len(self.vs))]

    def __repr__(self):
        return f"Waveform({len(self.vs)})"
