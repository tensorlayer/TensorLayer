
import sys
import testing
from yapf.yapflib.yapf_api import FormatCode



def _read_utf_8_file(filename):
    if sys.version_info.major == 2:
        return unicode(open(filename, 'rb').read(), 'utf-8')
    else:
        return open(filename, encoding='utf-8').read()


def check_all_files():
    for filename in testing.list_all_py_files():
        print(filename)
        code = _read_utf_8_file(filename)
        # https://pypi.python.org/pypi/yapf/0.20.2#example-as-a-module
        diff, changed = FormatCode(code, filename=filename, style_config='.style.yapf', print_diff=True)
        if changed:
            print(diff)
            yield filename


unformatted = list(check_all_files())

if unformatted:
    print('%d files need to be formatted, run the following commands to fix' % len(unformatted))
    for filename in unformatted:
        print('yapf -i %s' % filename)
    exit(1)
