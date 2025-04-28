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


def find_longest_zero_stretch(arr: list[int], wrap: bool) -> tuple[int, int]:
    if wrap: arr = arr + arr  # to handle wraparounds
    max_length = 0
    current_length = 0
    start_index = -1
    current_start = -1
    for i, num in enumerate(arr):
        if num == 0:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = current_start
        else:
            current_length = 0
    return start_index, max_length
