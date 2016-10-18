from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'tensorflow',
    'matplotlib',
]

setup(
    name = "tensorlayer",
    version = "1.2.2",
    include_package_data=True,
    author='TensorLayer Contributors',
    author_email='hao.dong11@imperial.ac.uk',
    url = "https://github.com/zsdonghao/tensorlayer" ,
    license = "apache" ,
    packages = find_packages(),
    install_requires=install_requires,
    # scripts=['tutorial_mnist.py'],
    description = "Deep learning and Reinforcement learning library for Researchers and Engineers",
    keywords = "deep learning, reinforcement learning, tensorflow",
    platform=['any'],
)
