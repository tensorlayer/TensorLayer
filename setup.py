from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'tensorflow',
    'matplotlib',
]

setup(
    name = "tensorlayer",
    version = "1.1",
    include_package_data=True,
    author='TensorLayer Team',
    author_email='hao.dong11@imperial.ac.uk',
    url = "https://github.com/zsdonghao/tensorlayer" ,
    license = "MIT" ,
    packages = find_packages(),
    install_requires=install_requires,
    scripts=['tutorial_mnist.py'],
    description = "A Deep learning and Reinforcement learning library for TensorFlow",
    keywords = "deep learning, reinforcement learning, tensorflow",
    platform=['any'],
)
