API - Natural Language Processing
==================================

Natural Language Processing and Word Representation.

.. automodule:: tensorlayer.nlp

.. autosummary::

   generate_skip_gram_batch

   sample
   sample_top

   SimpleVocabulary
   Vocabulary
   process_sentence
   create_vocab

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


Iteration function for training embedding matrix
-------------------------------------------------

.. autofunction:: generate_skip_gram_batch


Sampling functions
-------------------

.. autofunction:: sample
.. autofunction:: sample_top


Vector representations of words
-------------------------------

Vocabulary class
^^^^^^^^^^^^^^^^^

.. autoclass:: SimpleVocabulary
.. autoclass:: Vocabulary
.. autofunction:: process_sentence
.. autofunction:: create_vocab

Read words from file
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: simple_read_words
.. autofunction:: read_words


Read analogy question file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: read_analogies_file

Build vocabulary, word dictionary and word tokenization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: build_vocab
.. autofunction:: build_reverse_dictionary
.. autofunction:: build_words_dataset

Convert words to IDs and IDs to words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: words_to_word_ids
.. autofunction:: word_ids_to_words


Save vocabulary
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: save_vocab

Functions for translation
---------------------------

Word Tokenization
^^^^^^^^^^^^^^^^^^^

.. autofunction:: basic_tokenizer

Create or read vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: create_vocabulary
.. autofunction:: initialize_vocabulary

Convert words to IDs and IDs to words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sentence_to_token_ids
.. autofunction:: data_to_token_ids
