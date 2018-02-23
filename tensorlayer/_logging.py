import logging

logging.basicConfig(level=logging.INFO, format='[TL] %(message)s')


def info(fmt, *args):
    logging.info(fmt, *args)


def warning(fmt, *args):
    logging.warning(fmt, *args)
