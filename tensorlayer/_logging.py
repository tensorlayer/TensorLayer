import logging as _logger

logging=_logger.getLogger('tensorlayer')

_hander = _logger.StreamHandler()
formatter = _logger.Formatter('[TL] %(message)s')
_hander.setFormatter(formatter)
logging.addHandler(_hander)

logging.setLevel(_logger.INFO)


def info(fmt, *args):
    logging.info(fmt, *args)


def warning(fmt, *args):
    logging.warning(fmt, *args)
