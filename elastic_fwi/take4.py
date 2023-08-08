import inspect
import importlib
import re

def align(**kw):
    #required kw args
    s = kw['s']

    #optional kw args
    idt = kw.get('indent', 4*' ')
    idt_lvl = kw.get('indent_level', 1)
    cpl = kw.get('chars_per_line', 80)

    base_idt = idt_lvl * idt
    lng_idt = base_idt + idt
    N = cpl - len(base_idt)
    n = cpl - (len(idt) + len(base_idt))
    s = s.replace('\n',' ').strip().replace('  ', '')
    l = [f'{base_idt}{s[:N]}']
    for i in range((len(s) // n) + 1):
        l.append('%s%s'%(lng_idt, s[(N+i*n):(N+(i+1)*n)]))
    return '\n'.join(l)

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
    docstring_parts = [f'{base_indent}{cls.__name__}']
    if( stable ):
        docstring_parts.append(
            align(
                s=stable,
                indent=indent,
                indent_level=indent_level+1
            )
        )

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

    return ["\n".join(docstring_parts), og, line_no, stable]

def insert_docstring(**kw):
    meta = kw['meta']
    runner = kw['runner']

    if( len(meta) == 0 ): return runner
    else:
        s, dec, code_start, stable = meta.pop(0)
        dummy = [
            e for i,e in enumerate(runner) \
                    if i not in range(dec+1,code_start)
        ]
        dummy.insert(dec+1,'''"""\n***\n%s\n***\n%s\n"""\n'''%(stable, s))
        delta = code_start - dec
        for (i,e) in enumerate(meta):
            if( e[1] > code_start ): meta[i][1] -= delta
            if( e[2] > code_start ): meta[i][2] -= delta
        return insert_docstring(meta=meta, runner=dummy)
    
def process_module(module_name):
    module = importlib.import_module(module_name)
    src, top_line = inspect.getsourcelines(module)

    meta = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            meta.append(generate_docstring_for_class(obj))
    documented = ''.join(insert_docstring(meta=meta, runner=src))

    return documented

# Assuming the module name is 'my_module'
processed_module = process_module('elastic_class')

print(processed_module)

