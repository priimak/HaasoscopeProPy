from enum import Enum


class TriggerType(Enum):
    ZERO = 0
    ON_RISING_EDGE = 1
    ON_FALLING_EDGE = 2
    EXTERNAL = 3


class Commands:
    def get_waveform(self, num_samples: int) -> bytes:
        pass

    def arm_trigger(
            self,
            trigger_type: TriggerType,
            num_channels: int,
            oversample: bool,
            samples_after_trigger: int
    ) -> bytes:
        pass

    def get_version(self) -> int:
        """ Return firmware version. """
