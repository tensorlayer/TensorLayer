#!/usr/bin/env python
import os
import codecs

os.environ['TENSORLAYER_PACKAGE_BUILDING'] = 'True'

try:
    from setuptools import (
        setup,
        find_packages
    )

except ImportError:
    from distutils.core import (
        setup,
        find_packages
    )


from tensorlayer import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __version__
)


# =================== Reading Readme file as TXT files ===================

if os.path.exists('README.rst'):
    # codec is used for consistent encoding
    long_description = codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'),
        'r', 'utf-8'
    ).read()

else:
    long_description = 'See ' + __homepage__

# ======================= Reading Requirements files as TXT files =======================

def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]

# ======================= Defining the requirements var =======================



install_requires = req_file("requirements.txt")

extras_require = {
    'tf_cpu': req_file("requirements_tf_cpu.txt"),
    'tf_gpu': req_file("requirements_tf_gpu.txt"),
	'db': req_file("requirements_db.txt"),
	'dev': req_file("requirements_dev.txt"),
	'doc': req_file("requirements_doc.txt"),
	'extra': req_file("requirements_extra.txt"),
	'test': req_file("requirements_test.txt")
}

extras_require['all'] = sum([extras_require.get(key) for key in ['db', 'dev', 'doc', 'extra', 'test']], list())
extras_require['all_cpu'] = sum([extras_require.get(key) for key in ['all', 'tf_cpu']], list())
extras_require['all_gpu'] = sum([extras_require.get(key) for key in ['db', 'tf_gpu']], list())

# Readthedocs requires TF 1.5.0 to build properly
if os.environ.get('READTHEDOCS', None) == 'True':
    install_requires.append("tensorflow==1.5.0")

# ======================= Define the package setup =======================

setup(
    name=__package_name__,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description=__description__,
    long_description=long_description,

    # The project's main homepage.
    url=__repository_url__,
    download_url=__download_url__,

    # Author details
    author=__contact_names__,
    author_email=__contact_emails__,

    # maintainer Details
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,

    # The licence under which the project is released
    license=__license__,

    classifiers=[
        # How mature is this project? Common values are
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',

        # Indicate what your project relates to
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        # Additionnal Settings
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],

    keywords=__keywords__,
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # $ pip install -e .[test]
    extras_require=extras_require,
    scripts=[
        'tl',
    ],
)
