import os

from pydocstyle.checker import check
from pydocstyle.checker import violations

import testing

registry = violations.ErrorRegistry

_disabled_checks = [
    # 'D413',  # Missing blank line after last section
    # 'D205',  # 1 blank line required between summary line and description
    # 'D400',  # First line should end with a period
    # 'D107',  # Missing docstring in __init__
    # 'D401',  # First line should be in imperative mood
    # 'D210',  # No whitespaces allowed surrounding docstring text
    # 'D100',  # Missing docstring in public module
]


def check_all_files():
    for filename in testing.list_all_py_files():
        for err in check([filename]):
            if not err.code in _disabled_checks:
                yield err


def lookup_error_params(code):
    for group in registry.groups:
        for error_params in group.errors:
            if error_params.code == code:
                return error_params


violations = list(check_all_files())

if violations:
    counts = dict()
    for e in violations:
        print(e)
        counts[e.code] = counts.get(e.code, 0) + 1

    for n, code in sorted([(n, code) for code, n in counts.items()], reverse=True):
        p = lookup_error_params(code)
        print('%s %8d %s' % (code, n, p.short_desc))
    print('%s %8d violations' % ('tot', len(violations)))
    # TODO: exit(1)
