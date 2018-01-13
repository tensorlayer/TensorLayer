from setuptools import find_packages, setup

install_requires = [
    'numpy',
    # 'tensorflow', # user install it
    'scipy',
    'scikit-image',
    'matplotlib',
]

setup(
    name = "tensorlayer",
    version = "1.7.3",
    include_package_data=True,
    author='TensorLayer Contributors',
    author_email='hao.dong11@imperial.ac.uk',
    url = "https://github.com/zsdonghao/tensorlayer" ,
    license = "apache" ,
    packages = find_packages(),
    install_requires=install_requires,
    # scripts=['tutorial_mnist.py'],
    description = "Reinforcement Learning and Deep Learning Library for Researcher and Engineer.",
    keywords = "deep learning, reinforcement learning, tensorflow",
    platform=['any'],
)
