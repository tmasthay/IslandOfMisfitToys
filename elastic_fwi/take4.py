import inspect
import importlib
import re

def align(**kw):
    #required kw args
    s = kw['s']

    #optional kw args
    idt = kw.get('indent', 4*' ')
    idt_lvl = kw.get('indent_level', 1)

    base_idt = idt_lvl * idt
    return f'{base_idt}{s}'

def proc_inequal(**kw):
    #required kw args
    name = kw['name']
    left = kw['left']
    right = kw['right']

    if( left and ('<' not in left) and ('>' not in left) ): 
        left = left + '<='
    if( right and ('<' not in right) and ('>' not in right) ): 
        right = '<=' + right

    if( not (left + right) ): return ''

    proc_token = lambda v : re.sub(r'(<=?|>=?)', r' \1 ', v.replace(' ',''))
    return ''.join(
        [
            proc_token(left),
            name,
            proc_token(right)
        ]
    )

def ant_to_str(a, **kw):
    #required kw args
    name = kw['name']

    #optional kw args
    idt = kw.get('indent', 4 * ' ')
    idt_level = kw.get('indent_level', 1)

    base_idt = idt_level * idt
    sub_idt = base_idt + idt
    if( not a ): return f'{base_idt}{name} : NO ANNOTATION'
    if( type(a) == str ): return f'{base_idt}{name} : {a}'

    lst =  [re.sub(r"class|<|>|'", '', str(a.__origin__))] \
        + list(a.__metadata__)
    [lst.append('') for i in range(len(lst), 4)]
    lines = [f'{base_idt}{name}']
    if( lst[1] == '' ): lst[1] = 'NO HIGH-LEVEL ANNOTATION'
    lines.append(f'{sub_idt}{lst[1]}')
    lines.append(sub_idt + proc_inequal(name=name, left=lst[2], right=lst[3]))
    if( lines[-1] == name or lines[-1].replace(' ', '') == '' ): 
        lines = lines[:-1]
    for e in lst[4:]:
        lines.append(f'{base_idt}{e}')    
    return '\n'.join(lines)

def get_insertion_line(cls):
    lines, line_no = inspect.getsourcelines(cls)
    in_docstring = False
    just_exited_docstring = False
    entered_scope = False
    in_stable = False
    stable_docstring = ''

    for (idx,l) in enumerate(lines):
        if( not in_docstring and just_exited_docstring ): 
            just_exited_docstring = False
        if( not entered_scope and ':' in l ): 
            entered_scope = True
            continue
        sl = l.strip()
        if( (not sl) or (sl.startswith('#')) ): continue
        if( sl.startswith("'''") or sl.startswith('"""') ):
            in_docstring = not in_docstring
            if( not in_docstring ): just_exited_docstring = True
            continue
        if( in_docstring ):
            if( sl.startswith('***') ):
                in_stable = not in_stable
            if( in_stable ):
                stable_docstring += sl.replace('***','') + '\n'
            continue
        if( not just_exited_docstring ):
            if( in_stable ): 
                raise ValueError('Exited before closing stable docstring')
            return l, line_no, line_no + idx, stable_docstring
    
def generate_docstring_for_class(cls, **kw):
    indent = kw.get('indent', 4*' ')
    indent_level = kw.get('indent_level', 1)
    stable = kw.get('stable_docstring', '')
    base_indent = indent_level * indent

    try:
        slots = getattr(cls, "__slots__", [])
    except:
        slots = []
    
    insertion_line, og, line_no, stable = get_insertion_line(cls)
    input(f'{cls.__name__},{insertion_line},{og},{line_no},"{stable}"')
    docstring_parts = [f'{base_indent}{cls.__name__}']
    if( stable ):
        docstring_parts.append(f'{base_indent}{stable}')

    slot_idt = base_indent + indent
    docstring_parts.append(f'{slot_idt}Attributes')
    if( slots ):
        for slot in slots:
            ant = cls.__annotations__.get(
                slot, 
                'NO ANNOTATION'
            )
            docstring_parts.append(
                ant_to_str(
                    ant, 
                    name=slot, 
                    indent_level=indent_level+3
                )
            )
    else:
        docstring_parts.append(f'{slot_idt}{indent}NO SLOTS')

    input(docstring_parts)
    return "\n".join(docstring_parts)

def process_module(module_name):
    module = importlib.import_module(module_name)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            generated_docstring = generate_docstring_for_class(obj)
            obj.__doc__ = generated_docstring

    return module

# Assuming the module name is 'my_module'
processed_module = process_module('elastic_class')

# Print the modified classes' docstrings
for name, obj in inspect.getmembers(processed_module):
    if inspect.isclass(obj):
        print(f"Docstring for class {name}:")
        print(obj.__doc__)
        print("="*80)

