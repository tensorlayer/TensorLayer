#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sphinx.application import Sphinx


class DocTest(unittest.TestCase):
    source_dir = u'docs/'
    config_dir = u'docs/'
    output_dir = u'docs/build'
    doctree_dir = u'docs/build/doctrees'

    all_files = True

    def test_html_documentation(self):
        app = Sphinx(
            self.source_dir,
            self.config_dir,
            self.output_dir,
            self.doctree_dir,
            buildername='html',
            warningiserror=True,
        )
        app.build(force_all=self.all_files)
        # TODO: additional checks here if needed

    def test_text_documentation(self):
        # The same, but with different buildername
        app = Sphinx(
            self.source_dir,
            self.config_dir,
            self.output_dir,
            self.doctree_dir,
            buildername='text',
            warningiserror=False,
        )
        app.build(force_all=self.all_files)
        # TODO:  additional checks if needed

    def tearDown(self):
        # TODO: clean up the output directory
        pass


if __name__ == '__main__':
    unittest.main()
