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
   save_vocab

   words_to_word_ids
   word_ids_to_words

   basic_tokenizer
   create_vocabulary
   initialize_vocabulary
   sentence_to_token_ids
   data_to_token_ids

   moses_multi_bleu


Iteration function for training embedding matrix
-------------------------------------------------
.. autofunction:: generate_skip_gram_batch


Sampling functions
-------------------

Simple sampling
^^^^^^^^^^^^^^^^^^^
.. autofunction:: sample

Sampling from top k
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: sample_top

Vector representations of words
-------------------------------

Simple vocabulary class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SimpleVocabulary

Vocabulary class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Vocabulary

Process sentence
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: process_sentence

Create vocabulary
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: create_vocab

Read words from file
----------------------

Simple read file
^^^^^^^^^^^^^^^^^^
.. autofunction:: simple_read_words

Read file
^^^^^^^^^^^^^^^^^^
.. autofunction:: read_words


Read analogy question file
-----------------------------
.. autofunction:: read_analogies_file

Build vocabulary, word dictionary and word tokenization
--------------------------------------------------------

Build dictionary from word to id
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_vocab

Build dictionary from id to word
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_reverse_dictionary

Build dictionaries for id to word etc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: build_words_dataset

Save vocabulary
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_vocab


Convert words to IDs and IDs to words
--------------------------------------------------------
These functions can be done by ``Vocabulary`` class.

List of Words to IDs
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: words_to_word_ids

List of IDs to Words
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: word_ids_to_words



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


Metrics
---------------------------

BLEU
^^^^^^^^^^^^^^^^^^^
.. autofunction:: moses_multi_bleu
