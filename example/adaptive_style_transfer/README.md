## Adaptive Style Transfer in TensorFlow and TensorLayer

### Usage

1. TensorLayer implementation of the ICCV 2017 Paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) which supports any styles in one single model.

2. You can use the  <b>train.py</b> script to train your own model. To train the model, you need to download [MSCOCO dataset](http://cocodataset.org/#download) and [Wikiart dataset](https://www.kaggle.com/c/painter-by-numbers), and put the dataset images under the <b>'dataset/COCO\_train\_2014'</b> folder and <b>'dataset/wiki\_all\_images'</b> folder.


3. Alternatively, you can use the <b>test.py</b> script to run my pretrained models. My pretrained models can be downloaded from [here](https://github.com/tensorlayer/pretrained-models/tree/master/models/style_transfer_models_and_examples), and  should be put into the <b>'pretrained_models'</b> folder for testing.



### Results

Here are some result images (Left to Right: Content , Style , Result):

<div align="center">
   <img src="./images/content/content_1.png" width=250 height=250>
   <img src="./images/style/style_5.png" width=250 height=250>
   <img src="./images/output/style_5_content_1.jpg" width=250 height=250>
</div>


<div align="center">
   <img src="./images/content/content_2.png" width=250 height=250>
   <img src="./images/style/style11.png" width=250 height=250>
   <img src="./images/output/style_11_content2.png" width=250 height=250>
</div>

<div align="center">
   <img src="./images/content/chicago.jpg" width=250 height=250>
   <img src="./images/style/cat.jpg" width=250 height=250>
   <img src="./images/output/cat_chicago.jpg" width=250 height=250>
</div>



<div align="center">
   <img src="./images/content/lance.jpg" width=250 height=250>
   <img src="./images/style/lion.jpg" width=250 height=250>
   <img src="./images/output/lion_lance.jpg" width=250 height=250>
</div>
<div align="center">
   <img src="./images/content/content_4.png" width=250 height=250>
   <img src="./images/style/style_6.png" width=250 height=250>
   <img src="./images/output/style_6_content_4.jpg" width=250 height=250>
</div>

<div align="center">
   <img src="./images/content/lance.jpg" width=250 height=250>
   <img src="./images/style/udnie.jpg" width=250 height=250>
   <img src="./images/output/udnie_lance.jpg" width=250 height=250>
</div>

Enjoy!

### License

- This project for academic use only.
