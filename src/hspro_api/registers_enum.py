from enum import IntEnum


class RegisterIndex(IntEnum):
    ram_preoffset = 0
    ram_address_triggered = 1
    spistate = 2
    version = 3
    boardin = 4
    acqstate = 5
    eventcounter = 6
    sample_triggered = 7
    downsamplemergingcounter_triggered = 8
    downsamplemerging = 9
    downsample = 10
    highres = 11
