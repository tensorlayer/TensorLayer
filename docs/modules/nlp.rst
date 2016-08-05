:mod:`tunelayer.nlp`
======================

Natural Language Processing and Word Representation.

.. automodule:: tunelayer.nlp

.. autosummary::

   generate_skip_gram_batch

   sample
   sample_top

   simple_read_words
   read_words
   read_analogies_file
   build_vocab
   build_reverse_dictionary
   build_words_dataset
   words_to_word_ids
   word_ids_to_words
   save_vocab

   basic_tokenizer
   create_vocabulary
   initialize_vocabulary
   sentence_to_token_ids
   data_to_token_ids


Iteration functions
--------------------

.. autofunction:: generate_skip_gram_batch


Sampling functions
-------------------

.. autofunction:: sample
.. autofunction:: sample_top


Vector representations of words
-------------------------------

.. autofunction:: simple_read_words
.. autofunction:: read_words
.. autofunction:: read_analogies_file
.. autofunction:: build_vocab
.. autofunction:: build_reverse_dictionary
.. autofunction:: build_words_dataset
.. autofunction:: words_to_word_ids
.. autofunction:: word_ids_to_words
.. autofunction:: save_vocab

Functions for translation
---------------------------

.. autofunction:: basic_tokenizer
.. autofunction:: create_vocabulary
.. autofunction:: initialize_vocabulary
.. autofunction:: sentence_to_token_ids
.. autofunction:: data_to_token_ids
