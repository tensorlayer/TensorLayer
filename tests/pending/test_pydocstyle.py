#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tests.utils import list_all_py_files

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

        # TODO: fix all violations to make it empty
        _disabled_checks = [
            'D205',  # 1 blank line required between summary line and description
            'D102',  # Missing docstring in public method
            'D400',  # First line should end with a period
            'D100',  # Missing docstring in public module
            'D107',  # Missing docstring in __init__
            'D103',  # Missing docstring in public function
            'D401',  # First line should be in imperative mood
            'D101',  # Missing docstring in public class
            'D413',  # Missing blank line after last section
            'D105',  # Missing docstring in magic method
            'D104',  # Missing docstring in public package
            'D302',  # Use u"""for Unicode docstrings

            # Rules that has conflict with yapf
            'D202',  # No blank lines allowed after function docstring
        ]

        for filename in list_all_py_files():
            print(filename)
            for err in check([filename]):
                if not err.code in _disabled_checks:
                    cls.violations.append(err)

    def test_violations(self):
        if self.violations:
            counts = dict()

            for err in self.violations:
                counts[err.code] = counts.get(err.code, 0) + 1
                print(err)

            for n, code in sorted([(n, code) for code, n in counts.items()], reverse=True):
                p = lookup_error_params(code)
                print('%s %8d %s' % (code, n, p.short_desc))

            raise Exception('PyDoc Coding Style: %d violations have been found' % (len(self.violations)))


if __name__ == '__main__':
    unittest.main()
