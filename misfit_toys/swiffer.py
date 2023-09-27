import subprocess
from subprocess import check_output as co
from subprocess import CalledProcessError
from datetime import timedelta
import textwrap


def sco(s, split=True):
    try:
        u = co(s, shell=True).decode('utf-8')
        if split:
            return u.split('\n')[:-1]
        else:
            return u
    except CalledProcessError:
        return None


def sco_bash(function_name, *args, split=False):
    source_command = "source ~/.bash_functions"
    function_call = f'{function_name} {" ".join(map(str, args))}'
    full_command = f'{source_command} && {function_call}'

    # Invoke the bash shell and execute the command
    process = subprocess.Popen(
        ['bash', '-c', full_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    # Decode the output and error bytes to string
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    # If there's an error, raise it
    if stderr:
        raise RuntimeError(f"Error executing '{function_call}': {stderr}")

    while stdout[-1] == '\n':
        stdout = stdout[:-1]

    return stdout.split('\n') if split else stdout


def human_time(seconds, dec=2):
    s = str(timedelta(seconds=seconds))

    def clean_intra_day(u):
        if len(u.split(':')[0]) == 1:
            u = '0' + u
        if dec == 0 and '.' in u:
            u = u.split('.')[0]
        elif dec < 6 and '.' in u:
            u = u.split('.')[0] + '.' + u.split('.')[1][:dec]
        return u

    units = s.split(', ')
    units[-1] = clean_intra_day(units[-1])
    return ', '.join(units)


def see_fields(obj, *, field, member_paths, idt='    ', level=0):
    if member_paths is None:
        member_paths = []
    if len(member_paths) == 0:
        raise ValueError('member_paths must be a non-empty list of strings')
    if not isinstance(member_paths, list):
        raise ValueError('member_paths must be a list of strings')
    history = []
    for p in member_paths:
        p = p.split('.')
        c = obj
        s = ''
        for l, e in enumerate(p):
            if not hasattr(c, e):
                raise ValueError(
                    f'{c.__class__} does not have member {e}'
                    + f' at level {l} of path {p}'
                )
            c = getattr(c, e)
            s += f'{l*idt}{e}\n'
            if hasattr(c, field):
                s += f'{(l+1)*idt}{field}: {getattr(c, field)}\n'
        history.append(s)
    return '\n'.join(history)


def sub_dict(d, keys):
    return {k: v for k, v in d.items() if k in keys}


def istr(*args, idt_level=0, idt_str='    ', cpl=80):
    wrapper = textwrap.TextWrapper(width=cpl)
    s = ''.join(args)
    word_list = wrapper.wrap(text=s)
    base_idt = idt_str * idt_level
    full_idt = base_idt + idt_str
    res = base_idt + word_list[0]
    for line in word_list[1:]:
        res += '\n' + full_idt + line
    return res


def iprint(*args, idt_level=0, idt_str='    ', cpl=80, **kw):
    print(istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl), **kw)


def iraise(error_type, *args, idt_level=0, idt_str='    ', cpl=80):
    raise error_type(istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl))


def ireraise(e, *args, idt_level=0, idt_str='    ', cpl=80, idt_further=True):
    msg = str(e) + '\n'
    exception_type = type(e)
    full = istr(msg, idt_level=idt_level, idt_str=idt_str, cpl=cpl)
    if idt_further:
        idt_level += 1
    full += (
        '\n'
        + cpl * '*'
        + istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl)
        + cpl * '*'
    )
    raise exception_type(full)
