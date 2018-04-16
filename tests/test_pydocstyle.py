#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

try:
    import tests.testing as testing
except:
    import testing

from pydocstyle.checker import check
from pydocstyle.checker import violations

registry = violations.ErrorRegistry


def lookup_error_params(code):
    for group in registry.groups:
        for error_params in group.errors:
            if error_params.code == code:
                return error_params


class PyDOC_Style_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.violations = list()

        _disabled_checks = [
            'D202',  # No blank lines allowed after function docstring
            'D205',  # 1 blank line required between summary line and description
        ]

        for filename in testing.list_all_py_files():

            print(filename)
            for err in check([filename]):

                if not err.code in _disabled_checks:
                    cls.violations.append(err)

    def test_violations(self):
        if self.violations:

            counts = dict()

            for err in self.violations:
                counts[err.code] = counts.get(err.code, 0) + 1

            for n, code in sorted([(n, code) for code, n in counts.items()], reverse=True):
                p = lookup_error_params(code)
                print('%s %8d %s' % (code, n, p.short_desc))

            print()

            #raise Exception('PyDoc Coding Style: %d violations have been found' % ( len(self.violations)))  ## TODO: Correct these errors to allow Exception


if __name__ == '__main__':
    unittest.main()
