import argparse
import importlib.util
import inspect
import typing
import re

def prettify(s, indent='    ', indent_level=0, chars_per_line=80):
    if( s == '' ): return ''
    base_indent = indent_level * indent
    indent_chars = len(base_indent)
    block_len = chars_per_line - indent_chars
    start = s[:block_len]
    end = s[block_len:]
    block_len -= indent_chars
    t = [end[i*block_len:(i+1)*block_len] \
            for i in range(len(end) // block_len + 1)
    ]
    r = f'{base_indent}{start}'
    if( len(t[0]) > 0 ): r += f'\n{base_indent}{indent}'
    r += f'\n{base_indent}{indent}'.join(t)
    return r

def base_format(name, lst, indent='    ', indent_level=1, cpl=80): 
    base_indent = indent_level * indent
    s = f'{base_indent}{name} : {lst[0]}'
    ineql = lambda x : re.sub(r'\s*([\<\>]=?)\s*', r' \1 ', x)
    [lst.append('') for i in range(len(lst), 4)]
    if( lst[1] != '' ): 
        s += '\n' + prettify(lst[1], indent, indent_level+1, cpl)
    left, right = ineql(lst[2]), ineql(lst[3])
    if( len(left + right) > 0 ): 
        if( '<' not in left and len(left) > 0 ): left = left + ' <=  '
        if( '<' not in right and len(right) > 0 ): right = ' <= ' + right
        s += '\n' + prettify(f'{left}{name}{right}',
            indent,
            indent_level+1,
            cpl
        )
    for e in lst[4:]:
        s += '\n' + prettify(e, indent, indent_level+1, cpl)
    return s

def ant_to_str(name, 
    ant,
    indent='    ',
    indent_level=1,
    chars_per_line=80,
    proc=None
):
    if( not proc ): proc = base_format

    lst =  [re.sub(r"class|<|>|'", '', str(ant.__origin__))] \
        + list(ant.__metadata__)
    return proc(name, lst, indent, indent_level, chars_per_line)

def get_class_info(cls, 
    indent='    ', 
    indent_level=1, 
    chars_per_line=80,
    proc=None
):
    class_info = {
        "name": cls.__name__,
        "docstring": inspect.getdoc(cls),
        "attributes": [],
        "methods": {},
        "base classes": [base.__name__ for base in cls.__bases__],
        "slots": cls.__slots__
    }
    
    class_annotations = getattr(cls, '__annotations__', {})
    
    # extract info from class' __dict__
    for name, attr in cls.__dict__.items():
        if callable(attr):  
            class_info["methods"][name] = {"args": \
                list(inspect.signature(getattr(cls, name)).parameters.keys())
            }
        else:
            if name.startswith('__'): continue
            if name in class_annotations.keys():
                class_info["attributes"].append(
                    (
                        name, 
                        ant_to_str(name,
                            class_annotations[name],
                            indent,
                            indent_level,
                            chars_per_line,
                            proc
                        )
                    )
                )
            else:
                class_info["attributes"].append((name, "Not annotated"))

    return class_info

def generate_dict(**kw):
    spec = importlib.util.spec_from_file_location("module.name", kw['file'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class_members = inspect.getmembers(mod, inspect.isclass)

    data = {}
    for name, cls in class_members:
        if cls.__module__ == mod.__name__:
            data[name] = get_class_info(cls,
                indent=kw['indent'],
                indent_level=1,
                chars_per_line=kw['cpl'],
                proc=None
            )
    
    return data

def generate_docstring(**kw):
    d = generate_dict(**kw)
    print(d['Data']['slots'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="python file to inspect")
    parser.add_argument('--spaces', type=int, default=4, help="Indent spaces")
    parser.add_argument('--cpl', type=int, default=80, help='Chars per line')
    args = parser.parse_args()

    generate_docstring(file=args.file, 
        indent=args.spaces*' ', cpl=args.cpl)
