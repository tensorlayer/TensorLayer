import platform

if platform.system() != "Windows":
    import signal
else:
    signal = None

__all__ = ['TimeoutError', 'WindowsError', 'TimeoutContext']


class TimeoutError(Exception):
    pass


class WindowsError(Exception):
    pass


class TimeoutContext():
    """Timeout class using ALARM signal."""

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        if signal is None:
            raise WindowsError("Windows is not supported for this test")

        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise TimeoutError("A timeout error have been raised.")
