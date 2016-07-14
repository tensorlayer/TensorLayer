 # Copyright 2016 TensorLayer. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Text Generation using LSTM
---------------------------------

Link
------


"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import time



def main_how_to_use_embedding_layer():
    ## Step 1: Build the embedding matrix and load the existing embedding matrix.
    vocabulary_size = 50000
    embedding_size = 200
    model_file_name = "model_word2vec_50k_200"
    batch_size = None

    print("Load existing embedding matrix and dictionaries")
    all_var = tl.files.load_npy_to_any(name=model_file_name+'.npy')
    data = all_var['data']; count = all_var['count']
    dictionary = all_var['dictionary']
    reverse_dictionary = all_var['reverse_dictionary']

    tl.files.save_vocab(count, name='vocab_'+model_file_name+'.txt')

    del all_var, data, count

    load_params = tl.files.load_npz(name=model_file_name+'.npz')

    x = tf.placeholder(tf.int32, shape=[batch_size])
    y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])

    emb_net = tl.layers.EmbeddingInputlayer(
                    inputs = x,
                    vocabulary_size = vocabulary_size,
                    embedding_size = embedding_size,
                    name ='embedding_layer')

    sess.run(tf.initialize_all_variables())

    tl.files.assign_params(sess, [load_params[0]], emb_net)

    emb_net.print_params()
    emb_net.print_layers()

    ## Step 2: Input word(s), output the word vector(s).
    word = b'hello'
    word_id = dictionary[word]
    print('word_id:', word_id)

    words = [b'i', b'am', b'hao', b'dong']
    word_ids = tl.files.words_to_word_ids(words, dictionary)
    context = tl.files.word_ids_to_words(word_ids, reverse_dictionary)
    print('word_ids:', word_ids)
    print('context:', context)

    vector = sess.run(emb_net.outputs, feed_dict={x : [word_id]})
    print('vector:', vector.shape)

    vectors = sess.run(emb_net.outputs, feed_dict={x : word_ids})
    print('vectors:', vectors.shape)

def main_lstm_generate_text():
    pass


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    main_how_to_use_embedding_layer()
    # main_lstm_generate_text()














#
