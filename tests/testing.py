import os


def _list_py_files(root):
    for root, dirs, files in os.walk(root):
        if root.find('third_party') != -1:
            continue
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)


def list_all_py_files():
    dirs = ['tensorlayer', 'tests']
    for d in dirs:
        for filename in _list_py_files(d):
            yield filename
