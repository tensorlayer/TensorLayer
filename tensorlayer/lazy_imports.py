#! /usr/bin/python
# -*- coding: utf-8 -*-
""" This module provides lazy import functionality to improve the import
performance of nitime. For example, some parts of nitime leverage and import
matplotlib, which is quite a big package, yet most of the nitime code does not
depend on matplotlib. By lazily-loading a module, we defer the overhead of
importing it until the first time it is actually used, thereby speeding up
nitime imports.

A generic :class:`LazyImport` class is implemented which takes the module name
as a parameter, and acts as a proxy for that module, importing it only when
the module is used, but effectively acting as the module in every other way
(including inside IPython with respect to introspection and tab completion)
with the *exception* of reload() - reloading a :class:`LazyImport` raises an
:class:`ImportError`.

Commonly used nitime lazy imports are also defined in :mod:`nitime.lazy`, so
they can be reused throughout nitime.
"""
import sys
import types


class LazyImport(types.ModuleType):
    """
    This class takes the module name as a parameter, and acts as a proxy for
    that module, importing it only when the module is used, but effectively
    acting as the module in every other way (including inside IPython with
    respect to introspection and tab completion) with the *exception* of
    reload()- reloading a :class:`LazyImport` raises an :class:`ImportError`.

    >>> mlab = LazyImport('matplotlib.mlab')

    No import happens on the above line, until we do something like call an
    ``mlab`` method or try to do tab completion or introspection on ``mlab``
    in IPython.

    >>> mlab
    <module 'matplotlib.mlab' will be lazily loaded>

    Now the :class:`LazyImport` will do an actual import, and call the dist
    function of the imported module.

    >>> mlab.dist(1969,2011)
    42.0
    """

    def __getattribute__(self, x):
        # This method will be called only once, since we'll change
        # self.__class__ to LoadedLazyImport, and __getattribute__ will point
        # to module.__getattribute__

        name = object.__getattribute__(self, '__name__')
        __import__(name)

        # if name above is 'package.foo.bar', package is returned, the docs
        # recommend that in order to get back the full thing, that we import
        # and then lookup the full name is sys.modules, see:
        # http://docs.python.org/library/functions.html#__import__

        module = sys.modules[name]

        # Now that we've done the import, cutout the middleman and make self
        # act as the imported module

        class LoadedLazyImport(types.ModuleType):
            __getattribute__ = module.__getattribute__
            __repr__ = module.__repr__

        object.__setattr__(self, '__class__', LoadedLazyImport)

        # The next line will make "reload(l)" a silent no-op
        return module.__getattribute__(x)

    def __repr__(self):
        return "<module '%s' will be lazily loaded>" % object.__getattribute__(self, '__name__')


if 'sphinx' in sys.modules:
    lazy_doc = """
               WARNING: To get Sphinx documentation to build we disable
               LazyImports, which makes Sphinx incorrectly report this
               class as having a base class of object. In reality,
               :class:`LazyImport`'s base class is
               :class:`types.ModuleType`.
               """

    lazy_doc += LazyImport.__doc__

    class LazyImport(object):
        __doc__ = lazy_doc

        def __init__(self, x):
            __import__(x)
            self.module = sys.modules[x]

        def __getattr__(self, x):
            return self.module.__getattribute__(x)
