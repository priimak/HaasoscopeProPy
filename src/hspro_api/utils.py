def ensure_list[T](value: list[T] | None) -> list[T]:
    return [] if value is None else value


def getbit(i: int, n: int) -> int:
    """ Get bit n from byte i """
    return (i >> n) & 1


def bit_asserted(i: int, n: int) -> bool:
    """ True or False indicating if bit `n` is set in byte `i`. """
    return getbit(i, n) == 1


def int_to_bytes(int_value) -> list[int]:
    """ Convert length number to a 4-byte byte array (with type of 'bytes') """
    return [int_value & 0xff, (int_value >> 8) & 0xff, (int_value >> 16) & 0xff, (int_value >> 24) & 0xff]
