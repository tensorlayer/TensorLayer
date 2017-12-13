.. _more:

============
More
============


..
  Competitions
  ============

  Coming soon

FQA
===========

How to effectively learn TensorLayer
------------------------------------------
No matter what stage you are in, we recommend you to spend just 10 minutes to
read the source code of TensorLayer and the `Understand layer / Your layer <http://tensorlayer.readthedocs.io/en/stable/modules/layers.html>`_
in this website, you will find the abstract methods are very simple for everyone.
Reading the source codes helps you to better understand TensorFlow and allows
you to implement your own methods easily. For discussion, we recommend
`Gitter <https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>`_,
`Help Wanted Issues <https://waffle.io/zsdonghao/tensorlayer>`_,
`QQ group <https://github.com/zsdonghao/tensorlayer/blob/master/img/img_qq.png>`_
and `Wechat group <https://github.com/shorxp/tensorlayer-chinese/blob/master/docs/wechat_group.md>`_.

Beginner
^^^^^^^^^^^^^^
For people who new to deep learning, the contirbutors provided a number of tutorials in this website, these tutorials will guide you to understand autoencoder, convolutional neural network, recurrent neural network, word embedding and deep reinforcement learning and etc. If your already understand the basic of deep learning, we recommend you to skip the tutorials and read the example codes on `Github <https://github.com/zsdonghao/tensorlayer>`_ , then implement an example from scratch.

Engineer
^^^^^^^^^^^^^
For people from industry, the contirbutors provided mass format-consistent examples covering computer vision, natural language processing and reinforcement learning. Besides, there are also many TensorFlow users already implemented product-level examples including image captioning, semantic/instance segmentation, machine translation, chatbot and etc, which can be found online.
It is worth noting that a wrapper especially for computer vision `Tf-Slim <https://github.com/tensorflow/models/tree/master/slim#Pretrained>`_ can be connected with TensorLayer seamlessly.
Therefore, you may able to find the examples that can be used in your project.

Researcher
^^^^^^^^^^^^^
For people from academic, TensorLayer was originally developed by PhD students who facing issues with other libraries on implement novel algorithm. Installing TensorLayer in editable mode is recommended, so you can extend your methods in TensorLayer.
For researches related to image such as image captioning, visual QA and etc, you may find it is very helpful to use the existing `Tf-Slim pre-trained models <https://github.com/tensorflow/models/tree/master/slim#Pretrained>`_ with TensorLayer (a specially layer for connecting Tf-Slim is provided).


Exclude some layers from training
-----------------------------------

You may need to get the list of variables you want to update, TensorLayer provides two ways to get the variables list.

The first way is to use the all_params of a network, by default, it will store the variables in order.
You can print the variables information via
``tl.layers.print_all_variables(train_only=True)`` or ``network.print_params(details=False)``.
To choose which variables to update, you can do as below.

.. code-block:: python

  train_params = network.all_params[3:]

The second way is to get the variables by a given name. For example, if you want to get all variables which the layer name contain ``dense``, you can do as below.

.. code-block:: python

  train_params = tl.layers.get_variables_with_name('dense', train_only=True, printable=True)

After you get the variable list, you can define your optimizer like that so as to update only a part of the variables.

.. code-block:: python

  train_op = tf.train.AdamOptimizer(0.001).minimize(cost, var_list= train_params)


Visualization
--------------

Cannot Save Image
^^^^^^^^^^^^^^^^^^^^^^^

If you run the script via SSH control, sometime you may find the following error.

.. code-block:: bash

  _tkinter.TclError: no display name and no $DISPLAY environment variable

If happen, use ``import matplotlib`` and ``matplotlib.use('Agg')`` before ``import tensorlayer as tl``.
Alternatively, add the following code into the top of ``visualize.py`` or in your own code.

.. code-block:: python

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt


Install Master Version
-----------------------

To use all new features of TensorLayer, you need to install the master version from Github.
Before that, you need to make sure you already installed git.

.. code-block:: bash

  [stable version] pip install tensorlayer
  [master version] pip install git+https://github.com/zsdonghao/tensorlayer.git

Editable Mode
---------------

- 1. Download the TensorLayer folder from Github.
- 2. Before editing the TensorLayer ``.py`` file.

 - If your script and TensorLayer folder are in the same folder, when you edit the ``.py`` inside TensorLayer folder, your script can access the new features.
 - If your script and TensorLayer folder are not in the same folder, you need to run the following command in the folder contains ``setup.py`` before you edit ``.py`` inside TensorLayer folder.

  .. code-block:: bash

    pip install -e .


Load Model
--------------

Note that, the ``tl.files.load_npz()`` can only able to load the npz model saved by ``tl.files.save_npz()``.
If you have a model want to load into your TensorLayer network, you can first assign your parameters into a list in order,
then use ``tl.files.assign_params()`` to load the parameters into your TensorLayer model.





Recruitment
===========

TensorLayer Contributors
--------------------------

TensorLayer contributors are from Imperial College, Tsinghua University, Carnegie Mellon University, Google, Microsoft, Bloomberg and etc.
There are many functions need to be contributed such as
Maxout, Neural Turing Machine, Attention, TensorLayer Mobile and etc.
Please push on `GitHub`_, every bit helps and will be credited.
If you are interested in working with us, please
`contact us <hao.dong11@imperial.ac.uk>`_.


Data Science Institute, Imperial College London
------------------------------------------------

Data science is therefore by nature at the core of all modern transdisciplinary scientific activities, as it involves the whole life cycle of data, from acquisition and exploration to analysis and communication of the results. Data science is not only concerned with the tools and methods to obtain, manage and analyse data: it is also about extracting value from data and translating it from asset to product.

Launched on 1st April 2014, the Data Science Institute at Imperial College London aims to enhance Imperial's excellence in data-driven research across its faculties by fulfilling the following objectives.

The Data Science Institute is housed in purpose built facilities in the heart of the Imperial College campus in South Kensington. Such a central location provides excellent access to collabroators across the College and across London.

 - To act as a focal point for coordinating data science research at Imperial College by facilitating access to funding, engaging with global partners, and stimulating cross-disciplinary collaboration.
 - To develop data management and analysis technologies and services for supporting data driven research in the College.
 - To promote the training and education of the new generation of data scientist by developing and coordinating new degree courses, and conducting public outreach programmes on data science.
 - To advise College on data strategy and policy by providing world-class data science expertise.
 - To enable the translation of data science innovation by close collaboration with industry and supporting commercialization.

If you are interested in working with us, please check our
`vacancies <https://www.imperial.ac.uk/data-science/get-involved/vacancies/>`_
and other ways to
`get involved <https://www.imperial.ac.uk/data-science/get-involved/>`_
, or feel free to
`contact us <https://www.imperial.ac.uk/data-science/get-involved/contact-us/>`_.




.. _GitHub: https://github.com/zsdonghao/tensorlayer
.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
