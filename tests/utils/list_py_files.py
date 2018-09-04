import os

__all__ = [
    'list_all_py_files',
]

_excludes = [
    'tensorlayer/db.py',
]


def _list_py_files(root):
    for root, _dirs, files in os.walk(root):
        if root.find('third_party') != -1:
            continue
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)


def list_all_py_files():
    dirs = ['tensorlayer', 'tests', 'example']
    for d in dirs:
        for filename in _list_py_files(d):
            if filename not in _excludes:
                yield filename
