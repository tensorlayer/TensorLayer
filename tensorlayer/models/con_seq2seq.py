import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.models.seq2seq import Seq2seq
from tests.utils import CustomTestCase
from tensorlayer.cost import cross_entropy_seq

class Linear(Layer):
    def __init__(self, in_channels, out_channels, name=None):
        super(Linear, self).__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.build(None)
        self._built = True

    def build(self, inputs_shape):
        # W = [C, N]
        self.W = self._get_weights("W", shape=(self.in_channels, self.out_channels))

    def forward(self, inputs):
        # inputs = [B, H, C]
        # outputs = [B, H, N]
        outputs = tf.tensordot(inputs, self.W, axes=[2,0])
        return outputs




class Encoder(Model):

    def __init__(self, hidden_size, kernel_size, num_layers, embedding_layer, name=None):
        super(Encoder, self).__init__(name=name)
        self.vocab_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size
        self.hidden_size = hidden_size
        self.out_channels = hidden_size * 2
        self.kernel_size = kernel_size
        self.stride = 1
        self.layers = num_layers
        self.embedding = embedding_layer
        self.affine = Linear(in_channels=self.embedding_size, out_channels=self.hidden_size)
        self.conv = []
        for i in range(num_layers):
            self.conv.append(tl.layers.Conv1d(filter_size=self.kernel_size, 
                    n_filter=self.out_channels, 
                    stride=self.stride, in_channels=self.hidden_size))
        self.mapping = tl.layers.Dense(in_channels=self.hidden_size // 2, n_units=self.hidden_size)

    def forward(self, input):
        
        # batch, seq_len_src, dim
        inputs = self.embedding(input)
        batch_size = inputs.shape[0]
        # batch, seq_len_src, hidden
        outputs = self.affine(inputs)
        # short-cut
        # batch, seq_len_src, hidden
        _outputs = outputs

        for i in range(self.layers):
            # batch, seq_len_src, 2*hidden,
            outputs = self.conv[i](outputs)
            # batch, seq_len_src, hidden
            # Gated Linear unit function
            outputs = tf.math.multiply(outputs[:,:,:self.hidden_size], tf.sigmoid(outputs[:,:,self.hidden_size:]))
            # A, B: batch, seq_len_src, hidden / 2
            A, B = outputs[:,:,:self.hidden_size//2], outputs[:,:,self.hidden_size//2:]
            # A2: batch * seq_len_src, hidden / 2
            A2 = tf.reshape(A, [-1,A.shape[-1]])
            # B2: batch * seq_len_src, hidden / 2
            B2 = tf.reshape(B, [-1,B.shape[-1]])
            # attn: batch * seq_len_src, hidden / 2
            attn = A2 * tf.nn.softmax(B2)
            _attn = tf.reshape(attn, [batch_size, -1, self.hidden_size//2])
            # attn2: batch * seq_len_src, hidden
            attn2 = self.mapping(attn)
            # outputs: batch, seq_len_src, hidden
            outputs = tf.reshape(attn2, [batch_size, -1, self.hidden_size])
            # batch, seq_len_src, hidden_size
            _outputs = outputs + _outputs
            
        
        return _attn, _outputs


class Decoder(Model):

    def __init__(self, hidden_size, embedding_layer, kernel_size, num_layers, name=None):
        super(Decoder, self).__init__(name=name)

        self.vocab_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size
        self.hidden_size = hidden_size

        self.in_channels = hidden_size
        self.out_channels = hidden_size * 2
        self.kernel_size = kernel_size
        self.stride = 1
        self.layers = num_layers

        self.embedding = embedding_layer
        self.affine = Linear(self.embedding_size, self.hidden_size)
        self.conv = []
        for i in range(num_layers):
            self.conv.append(tl.layers.Conv1d(n_filter=self.out_channels, in_channels=self.in_channels, 
                filter_size=kernel_size, stride=self.stride)
                )
        self.mapping = tl.layers.Dense(in_channels=self.hidden_size // 2, n_units=self.hidden_size)
        self.fc = Linear(self.hidden_size, self.vocab_size)


    # enc_attn: src_seq_len, hidden_size
    def forward(self, inputs):
        
        target = inputs[0]
        enc_attn = inputs[1]
        source_seq_out = inputs[2]

        # batch, seq_len_tgt, dim
        inputs = self.embedding(target)
        batch_size = inputs.shape[0]
        # batch, seq_len_tgt, hidden
        outputs = self.affine(inputs)

        for i in range(self.layers):

            # This is the residual connection,
            # for the output of the conv will add kernel_size / 2 elements
            # before and after the origin input
            # if i > -1:
            #     conv_out = conv_out + outputs

            # batch, seq_len_src, 2*hidden,
            outputs = self.conv[i](outputs)
            # batch, seq_len_src, hidden
            # Gated Linear unit function
            outputs = tf.math.multiply(outputs[:,:,:self.hidden_size], tf.sigmoid(outputs[:,:,self.hidden_size:]))
            # A, B: batch, seq_len_src, hidden / 2
            A, B = outputs[:,:,:self.hidden_size//2], outputs[:,:,self.hidden_size//2:]
            # A2: batch * seq_len_src, hidden / 2
            A2 = tf.reshape(A, [-1,A.shape[-1]])
            # B2: batch * seq_len_src, hidden / 2
            B2 = tf.reshape(B, [-1,B.shape[-1]])
            # attn: batch * seq_len_src, hidden / 2
            dec_attn = A2 * tf.nn.softmax(B2)
            # attn2: batch * seq_len_src, hidden
            dec_attn2 = self.mapping(dec_attn)
            dec_attn2 = tf.reshape(dec_attn2, [batch_size, -1, self.hidden_size])


            # dec_attn: batch, seq_len_tgt, hidden_size//2
            dec_attn = tf.reshape(dec_attn, [batch_size, -1, self.hidden_size//2])
            
            # enc_attn: batch, seq_len_src, hidden_size//2
            # dec_atten: batch, seq_len_tgt, hidden_size//2
            # attn_matrix: batch, seq_len_tgt, seq_len_src
            enc_attn = tf.transpose(enc_attn, perm=[0,2,1])
            _attn_matrix = tf.matmul(dec_attn, enc_attn)
            enc_attn = tf.transpose(enc_attn, perm=[0,2,1])
            attn_matrix = tf.nn.softmax(_attn_matrix, axis=-1)


            # attns: batch, seq_len_tgt, hidden_size
            attns = tf.matmul(attn_matrix, source_seq_out)

            # outpus: batch, seq_len_tgt, hidden_size
            outputs = dec_attn2 + attns

        # outpus: batch, seq_len_tgt, vocab_size
        outputs = tf.nn.softmax(self.fc(outputs))

        return outputs


class ConvSeq2Seq(Model):
    def __init__(self, encoder, decoder, name=None):
        super(ConvSeq2Seq, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, inputs):
        # attn: batch, seq_len, hidden
        # out: batch, seq_len, hidden_size
        source, target = inputs[0], inputs[1]
        attn, source_seq_out = self.encoder(source)

        # batch, seq_len_tgt, vocab_size
        out = self.decoder([target, attn, source_seq_out])

        return out
        
# enc_attn = tl.layers.Input((16,5,64))
# src_out = tl.layers.Input((16,5,128))
# input = tl.layers.Input((16,5), dtype=tf.int32)
# model_ = ConvSeq2Seq(
#     decoder = Decoder(hidden_size=128, kernel_size=3, num_layers=2, 
#     embedding_layer=tl.layers.Embedding(vocabulary_size=500,embedding_size=50)),
#     encoder = Encoder(hidden_size=128, kernel_size=3, num_layers=2, 
#     embedding_layer=tl.layers.Embedding(vocabulary_size=500,embedding_size=50))
# )

# model_.train()
# print(model_)


class Model_SEQ2SEQ_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.batch_size = 16

        cls.vocab_size = 20
        cls.embedding_size = 32
        cls.dec_seq_length = 5
        cls.trainX = np.random.randint(20, size=(50, 5))
        cls.trainY = np.random.randint(20, size=(50, cls.dec_seq_length + 1))
        cls.trainY[:, 0] = 0  # start_token == 0

        # Parameters
        cls.src_len = len(cls.trainX)
        cls.tgt_len = len(cls.trainY)

        assert cls.src_len == cls.tgt_len

        cls.num_epochs = 500
        cls.n_step = cls.src_len // cls.batch_size

    @classmethod
    def tearDownClass(cls):
        pass

    def test_basic_simpleSeq2Seq(self):
        model_ = ConvSeq2Seq(
            decoder = Decoder(hidden_size=128, kernel_size=3, num_layers=5, 
            embedding_layer=tl.layers.Embedding(vocabulary_size=self.vocab_size,embedding_size=self.embedding_size)),
            encoder = Encoder(hidden_size=128, kernel_size=3, num_layers=5, 
            embedding_layer=tl.layers.Embedding(vocabulary_size=self.vocab_size,embedding_size=self.embedding_size))
        )

        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        for epoch in range(self.num_epochs):
            model_.train()
            
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                dec_seq = Y[:, :-1]
                target_seq = Y[:, 1:]

                with tf.GradientTape() as tape:
                    ## compute outputs
                    output = model_(inputs=[X, dec_seq])

                    output = tf.reshape(output, [-1, self.vocab_size])
                    loss = cross_entropy_seq(logits=output, target_seqs=target_seq)
                    
                    grad = tape.gradient(loss, model_.all_weights)
                    optimizer.apply_gradients(zip(grad, model_.all_weights))

                total_loss += loss
                n_iter += 1

            model_.eval()
            test_sample = trainX[0:2, :].tolist()

            top_n = 1
            # for i in range(top_n):
            #     prediction = model_([test_sample], seq_length=self.dec_seq_length, start_token=0, top_n=1)
            #     print("Prediction: >>>>>  ", prediction, "\n Target: >>>>>  ", trainY[0:2, 1:], "\n\n")

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()




