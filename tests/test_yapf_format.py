#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest

try:
    import tests.testing as testing
except:
    import testing

from yapf.yapflib.yapf_api import FormatCode


def _read_utf_8_file(filename):
    if sys.version_info.major == 2:  ## Python 2 specific
        with open(filename, 'rb') as f:
            return unicode(f.read(), 'utf-8')
    else:
        with open(filename, encoding='utf-8') as f:
            return f.read()


class YAPF_Style_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.files_badly_formated = list()

        for filename in testing.list_all_py_files():

            print(filename)
            code = _read_utf_8_file(filename)

            # https://pypi.python.org/pypi/yapf/0.20.2#example-as-a-module
            diff, changed = FormatCode(code, filename=filename, style_config='.style.yapf', print_diff=True)

            if changed:
                print(diff)
                cls.files_badly_formated.append(filename)

    def test_unformated_files(self):
        if self.files_badly_formated:
            print()

            for filename in self.files_badly_formated:
                print('yapf -i %s' % filename)

            raise Exception("Bad Coding Style: %d files need to be formatted, run the following commands to fix" % len(self.files_badly_formated))


if __name__ == '__main__':
    unittest.main()
