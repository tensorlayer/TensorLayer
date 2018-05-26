import argparse
import requests
import logging

import pip._internal

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get the nth version of a given package')
    parser.add_argument('--package', type=str, required=True, help='The PyPI you want to inspect')
    parser.add_argument('--nth_last_version', type=int, default=1, help='The nth last package will be retrieved')
    parser.add_argument('--prerelease', help='Get PreRelease Package Version', action='store_true')
    parser.add_argument('--debug', help='Print debug information', action='store_true')

    args = parser.parse_args()

    # create logger
    logger = logging.getLogger("PyPI_CLI")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("Package: %s" % args.package)
    logger.debug("nth_last_version: %s" % args.nth_last_version)
    logger.debug("prerelease: %s" % args.prerelease)
    logger.debug("debug: %s" % args.debug)

    finder = pip._internal.index.PackageFinder(
        [],
        ['https://pypi.python.org/simple'],
        session=requests.Session()
    )
    results = finder.find_all_candidates(args.package)
    tmp_versions = [str(p.version) for p in results]

    logger.debug("%s" % tmp_versions)

    versions = list()
    for el in tmp_versions:
        if el not in versions:
            versions.append(el)

    pos = -1
    nth_version = 1

    while True:
        fetched_version = versions[pos]

        logger.debug("Version: %s" % fetched_version)

        if nth_version == args.nth_last_version:
            if args.prerelease or not ("rc" in fetched_version or "a" in fetched_version or "b" in fetched_version):
                break
            else:
                pos -= 1
                continue

        pos -= 1
        nth_version += 1

    print(fetched_version)
