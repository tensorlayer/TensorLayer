
### Introduction

The demos implement [FastText](http://arxiv.org/abs/1607.01759)[1] for sentence classification. 

Code: [tutorial_imdb_fasttext.py](tutorial_imdb_fasttext.py)

FastText is a simple model for text classification with performance often close
to state-of-the-art, and is useful as a solid baseline.

There are some important differences between this implementation and what
is described in the paper. Instead of Hogwild! SGD[2], we use Adam optimizer
with mini-batches. Hierarchical softmax is also not supported; if you have
a large label space, consider utilizing candidate sampling methods provided
by TensorFlow[3].

After 5 epochs, you should get test accuracy around 90.3%.

### References

[1] Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016).
    Bag of Tricks for Efficient Text Classification.
    <http://arxiv.org/abs/1607.01759>

[2] Recht, B., Re, C., Wright, S., & Niu, F. (2011).
    Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    In Advances in Neural Information Processing Systems 24 (pp. 693–701).

[3] <https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling>
