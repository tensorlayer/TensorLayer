from setuptools import find_packages, setup

install_requires = [
    'numpy',
    # 'tensorflow', # user install it
    'scipy',
    'scikit-image',
    'matplotlib',
]

setup(
    name="tensorlayer",
    version="1.8.0",
    include_package_data=True,
    author='TensorLayer Contributors',
    author_email='hao.dong11@imperial.ac.uk',
    url="https://github.com/tensorlayer/tensorlayer",
    license="apache",
    packages=find_packages(),
    install_requires=install_requires,
    scripts=[
        'tl',
    ],
    description="Reinforcement Learning and Deep Learning Library for Researcher and Engineer.",
    keywords="deep learning, reinforcement learning, tensorflow",
    platform=['any'],
)
