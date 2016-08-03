from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'tensorflow',
    'matplotlib',
]

setup(
    name = "tunelayer",
    version = "1.1",
    include_package_data=True,
    author='TuneLayer Contributors',
    author_email='hao.dong11@imperial.ac.uk',
    url = "https://github.com/zsdonghao/tunelayer" ,
    license = "apache" ,
    packages = find_packages(),
    install_requires=install_requires,
    # scripts=['tutorial_mnist.py'],
    description = "Deep learning and Reinforcement learning library for Researchers and Engineers",
    keywords = "deep learning, reinforcement learning, tensorflow",
    platform=['any'],
)
