# post-processing for Part-Affinity Fields Map implemented in C++ & Swig

Need to install swig.

```bash
$ sudo apt install swig
```

You need to build pafprocess module which is written in c++. It will be used for post processing.

```bash
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

