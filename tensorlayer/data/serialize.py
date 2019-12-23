import msgpack_numpy

MAX_MSGPACK_LEN = 1000000000


def convert_to_bytes(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
    """
    return msgpack_numpy.dumps(obj, use_bin_type=True)


def load_from_bytes(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    # Since 0.6, the default max size was set to 1MB.
    # We change it to approximately 1G.
    return msgpack_numpy.loads(buf, raw=False,
                               max_bin_len=MAX_MSGPACK_LEN,
                               max_array_len=MAX_MSGPACK_LEN,
                               max_map_len=MAX_MSGPACK_LEN,
                               max_str_len=MAX_MSGPACK_LEN)
