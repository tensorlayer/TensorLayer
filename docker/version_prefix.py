import argparse
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Determine the version prefix to apply depending on the version name')

    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help='The Package Version to be installed in the container'
    )

    parser.add_argument('--debug', help='Print debug information', action='store_true')

    args = parser.parse_args()

    # create logger
    logger = logging.getLogger("VERSION_PREFIX_CLI")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("Package Version: %s" % args.version)

    if "rc" in args.version or "a" in args.version or "b" in args.version:
        print("latest-dev")
    else:
        print("latest")
