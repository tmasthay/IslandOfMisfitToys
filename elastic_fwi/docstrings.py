import argparse
import importlib.util
import inspect
import typing
import re

def prettify(**kw):
    #required keywords
    s = kw['s']

    #optional keywords
    indent = kw.get('indent', 4*' ')
    indent_level = kw.get('indent_level', 0)
    chars_per_line = kw.get('chars_per_line', 80)

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

def base_format(**kw):
    #required keywords
    name = kw['name']
    lst = kw['lst']

    #optional keywords
    indent = kw.get('indent', 4 * ' ')
    indent_level = kw.get('indent_level', 1)
    cpl = kw.get('chars_per_line', 80) 

    base_indent = indent_level * indent
    s = f'{base_indent}{name} : {lst[0]}'
    ineql = lambda x : re.sub(r'\s*([\<\>]=?)\s*', r' \1 ', x)
    [lst.append('') for i in range(len(lst), 4)]
    if( lst[1] != '' ): 
        s += '\n' + prettify(s=lst[1], 
            indent=indent, 
            indent_level=indent_level+1, 
            chars_per_line=cpl
        )
    left, right = ineql(lst[2]), ineql(lst[3])
    if( len(left + right) > 0 ): 
        if( '<' not in left and len(left) > 0 ): left = left + ' <=  '
        if( '<' not in right and len(right) > 0 ): right = ' <= ' + right
        s += '\n' + prettify(s=f'{left}{name}{right}',
            indent=indent,
            indent_level=indent_level+1,
            chars_per_line=cpl
        )
    for e in lst[4:]:
        s += '\n' + prettify(e, indent, indent_level+1, cpl)
    return s 

def ant_to_str(**kw):
    #required keywords
    name = kw['name']
    ant = kw['ant']

    #optional keywords
    indent = kw.get('indent', 4 * ' ')
    indent_level = kw.get('indent_level', 1)
    chars_per_line = kw.get('chars_per_line', 80)
    proc = kw.get('proc', None)

    if( not proc ): proc = base_format
    if( not ant ): return f'{indent_level*indent}{name}: NO ANNOTATION'

    lst =  [re.sub(r"class|<|>|'", '', str(ant.__origin__))] \
        + list(ant.__metadata__)
    return proc(name=name, 
        lst=lst, 
        indent=indent, 
        indent_level=indent_level, 
        chars_per_line=chars_per_line
    )

def proc_ant(v, return_ant):
    if( return_ant ):
        return v.return_annotation if v.return_annotation != v.empty else None
    else:
        return v.annotation if v.annotation != v.empty else None
    
def header(**kw):
    head = kw['head']

    indent = kw.get('indent', 4*' ')
    indent_level = kw.get('indent_level', 1)
    c = kw.get('c', '-')

    base_idt = indent_level * indent
    s = f'{base_idt}{head}\n'
    if( c ): s += f'{base_idt}{len(head)*c}\n'
    return s

def func_docstring(d, **kw): 
    params = d['params']
    return_ant = d['return_annotation']
    name = d['name']
    param_head = d.get('param_head', 'Parameters')
    return_head  = d.get('return_head', 'Return Value')

    indent = kw.get('indent', 4 * ' ')
    indent_level = kw.get('indent_level', 2)
    chars_per_line = kw.get('chars_per_line', 80)
    proc = kw.get('proc', None)
    c = kw.get('c', '-')
    postpend = kw.get('postpend', '\n')

    base_idt = indent_level * indent
    sub_idt = base_idt + indent
    s = header(head=name, 
        indent=indent, 
        indent_level=indent_level, 
        c=c
    )

    param_head = 'Parameters'
    s += header(head=param_head, 
        indent=indent, 
        indent_level=indent_level+1, 
        c=c
    )
    if( params ):
        for k,v in params.items():
            s += ant_to_str(name=k, 
                ant=v, 
                indent=indent, 
                indent_level=indent_level+2, 
                chars_per_line=chars_per_line, 
                proc=proc
            )
            s += '\n'
    else:
        s += sub_idt + 'None'
    
    s += header(head=return_head, 
        indent=indent, 
        indent_level=indent_level+1, 
        chars_per_line=chars_per_line
    )
    if( return_ant ):
        for k,v in return_ant.items():
            s += ant_to_str(name=k, 
                ant=v, 
                indent=indent, 
                indent_level=indent_level+1, 
                chars_per_line=chars_per_line, 
                proc=proc
            )
            s += '\n'
    else:
        s += '\n' + base_idt + 2 * indent + 'None'
    return s + postpend

def get_class_info(cls, 
    indent=4*' ',
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
            sig = inspect.signature(attr) 
            params = sig.parameters
            class_info["methods"][name] = {'params': {},
                'return_annotation': proc_ant(sig, True)
            }
            for param_name, param in params.items():
                class_info['methods'][name]['params'][param_name] = \
                    proc_ant(param, False)
        else:
            if name.startswith('__'): continue
            if name in class_annotations.keys():
                class_info["attributes"].append(
                    (
                        name, 
                        ant_to_str(name=name,
                            ant=class_annotations[name],
                            indent=indent,
                            indent_level=indent_level,
                            chars_per_line=chars_per_line,
                            proc=proc
                        )
                    )
                )
            else:
                class_info["attributes"].append((name, "NOT ANNOTATED"))

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
    classes = dict()
    for k, v in d.items():
        classes[k] = ('' if not v['docstring'] else v['docstring']) + '\n'
        classes[k] += header(head='Attributes',
            indent=kw.get('indent', 4*' '),
            indent_level=kw.get('indent_level', 1),
            c=kw.get('c', '-')
        )
        classes[k] += '\n'.join(
            [
                e[1] for e in v['attributes'] \
                    if not e[0].startswith('_')
            ]
        ) + '\n\n'
        classes[k] += header(head='Member Functions',
            indent=kw.get('indent', 4*' '),
            indent_level=kw.get('indent_level', 1),
            c=kw.get('c', '-')
        ) 
        for k_hat, v_hat in v['methods'].items():
            classes[k] += func_docstring({'name': k_hat, **v_hat}, **kw)
    return classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="python file to inspect")
    parser.add_argument('--spaces', type=int, default=4, help="Indent spaces")
    parser.add_argument('--cpl', type=int, default=80, help='Chars per line')
    args = parser.parse_args()

    d = generate_docstring(file=args.file, 
        indent=args.spaces*' ', cpl=args.cpl)
    
    for k,v in d.items():
        print(v)
