from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorlayer.models.transformer import Transformer
from tensorlayer.models.transformer.utils import metrics
from tensorlayer.models.transformer.utils import attention_visualisation
import tensorlayer as tl
""" Translation from Portugese to English by Transformer model
This tutorial provides basic instructions on how to define and train Transformer model on Tensorlayer for 
Translation task. You can also learn how to visualize the attention block via this tutorial. 
"""


def set_up_dataset():
    # Set up dataset for Portugese-English translation from the TED Talks Open Translation Project.
    # This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.
    # https://www.ted.com/participate/translate

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    # Set up tokenizer and save the tokenizer
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() and pt.numpy() for pt, en in train_examples), target_vocab_size=2**14
    )

    tokenizer.save_to_file("tokenizer")
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer")

    return tokenizer, train_examples


def test_tokenizer_success(tokenizer):
    sample_string = 'TensorLayer is awesome.'

    tokenized_string = tokenizer.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer.decode(tokenized_string)
    print('The original string: {}'.format(original_string))
    assert original_string == sample_string


def generate_training_dataset(train_examples, tokenizer):

    def encode(lang1, lang2):
        lang1 = tokenizer.encode(lang1.numpy()) + [tokenizer.vocab_size + 1]

        lang2 = tokenizer.encode(lang2.numpy()) + [tokenizer.vocab_size + 1]

        return lang1, lang2

    MAX_LENGTH = 50

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset


def model_setup(tokenizer):
    # define Hyper parameters for transformer
    class HYPER_PARAMS(object):
        vocab_size = tokenizer.vocab_size + 10
        encoder_num_layers = 4
        decoder_num_layers = 4
        hidden_size = 128
        ff_size = 512
        num_heads = 8
        keep_prob = 0.9

        # Default prediction params
        extra_decode_length = 50
        beam_size = 5
        alpha = 0.6  # used to calculate length normalization in beam search

        label_smoothing = 0.1
        learning_rate = 2.0
        learning_rate_decay_rate = 1.0
        learning_rate_warmup_steps = 4000

        sos_id = 0
        eos_id = tokenizer.vocab_size + 1

    model = Transformer(HYPER_PARAMS)

    # Set the optimizer
    learning_rate = CustomSchedule(HYPER_PARAMS.hidden_size, warmup_steps=HYPER_PARAMS.learning_rate_warmup_steps)
    optimizer = tl.optimizers.LazyAdamOptimizer(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return model, optimizer, HYPER_PARAMS


# Use the Adam optimizer with a custom learning rate scheduler according to the formula in the Paper "Attention is All you need"
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=5):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def tutorial_transformer():
    tokenizer, train_examples = set_up_dataset()
    train_dataset = generate_training_dataset(train_examples, tokenizer)
    model, optimizer, HYPER_PARAMS = model_setup(tokenizer)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for (batch, (inp, tar)) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits, weights_encoder, weights_decoder = model(inputs=inp, targets=tar)
                logits = metrics.MetricLayer(HYPER_PARAMS.vocab_size)([logits, tar])
                logits, loss = metrics.LossLayer(HYPER_PARAMS.vocab_size, 0.1)([logits, tar])
                grad = tape.gradient(loss, model.all_weights)
                optimizer.apply_gradients(zip(grad, model.all_weights))
                if (batch % 50 == 0):
                    print('Batch ID {} at Epoch [{}/{}]: loss {:.4f}'.format(batch, epoch + 1, num_epochs, loss))

    model.eval()
    sentence_en = tokenizer.encode('TensorLayer is awesome.')
    [prediction, weights_decoder], weights_encoder = model(inputs=[sentence_en])

    predicted_sentence = tokenizer.decode([i for i in prediction["outputs"][0] if i < tokenizer.vocab_size])
    print("Translated: ", predicted_sentence)

    # visualize the self attention
    tokenizer_str = [tokenizer.decode([ts]) for ts in (sentence_en)]
    attention_visualisation.plot_attention_weights(weights_encoder["layer_0"], tokenizer_str, tokenizer_str)


if __name__ == "__main__":
    tutorial_transformer()
