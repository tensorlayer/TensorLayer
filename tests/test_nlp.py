#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase
import nltk
nltk.download('punkt')

class Test_Leaky_ReLUs(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_generate_skip_gram_batch(self):
        data = [1,2,3,4,5,6,7,8,9,10,11]
        batch, labels, data_index = tl.nlp.generate_skip_gram_batch(data=data, batch_size=8, num_skips=2, skip_window=1, data_index=0)
        print(batch)
        print(labels)

    def test_process_sentence(self):
        c = "how are you?"
        c = tl.nlp.process_sentence(c)
        print(c)

    def test_words_to_word_id(self):
        words = tl.files.load_matt_mahoney_text8_dataset()
        vocabulary_size = 50000
        data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size, True)
        ids = tl.nlp.words_to_word_ids(words, dictionary)
        context = tl.nlp.word_ids_to_words(ids, reverse_dictionary)
        # print(ids)
        # print(context)


if __name__ == '__main__':

    unittest.main()
