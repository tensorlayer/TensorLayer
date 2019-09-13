import matplotlib.pyplot as plt
import tensorflow as tf


def plot_attention_weights(attention, key, query):
    '''Attention visualisation for Transformer

    Parameters
    ----------
    attention : attention weights
        shape of (1, number of head, length of key, length of query).
    
    key : key for attention computation
        a list of values which would be shown as xtick labels

    value : value for attention computation
        a list of values which would be shown as ytick labels

    '''

    fig = plt.figure(figsize=(16, 8))
    attention = tf.squeeze(attention, axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(attention.shape[0] // 2, 2, head + 1)
        ax.matshow(attention[head], cmap='viridis')
        fontdict = {'fontsize': 12}
        ax.set_xticks(range(len(key)))
        ax.set_yticks(range(len(query)))

        # ax.set_ylim(len(query)-1.5, -0.5)
        ax.set_xticklabels([str(i) for i in key], fontdict=fontdict, rotation=90)

        ax.set_yticklabels([str(i) for i in query], fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1), fontdict=fontdict)
    plt.tight_layout()
    plt.show()
