"""
Collection of miscellaneous helper functions to clean up code, like a Swiffer.

Functions:
    sco: Execute a shell command and return the output.
    sco_bash: Execute a bash function and return the output.
    human_time: Convert seconds to a human-readable time format.
    see_fields: Get the values of specified fields in an object.
    sub_dict: Create a new dictionary with only the specified keys.
    istr: Indent and wrap a string.
    iprint: Print an indented and wrapped string.
    iraise: Raise an exception with an indented and wrapped error message.
    ireraise: Re-raise an exception with an indented and wrapped error message.
"""

import subprocess
import textwrap
from datetime import timedelta
from subprocess import CalledProcessError
from subprocess import check_output as co
import sys


def sco(s, split=True):
    """
    Execute a shell command and return the output.

    Args:
        s (str): The shell command to execute.
        split (bool, optional): Whether to split the output by lines. Defaults to True.

    Returns:
        str or list: The output of the shell command. If split is True, returns a list of lines, otherwise returns a single string.
    """
    try:
        u = co(s, shell=True).decode("utf-8")
        if split:
            return u.strip().split("\n")
        else:
            return u.strip()
    except CalledProcessError:
        return None


def sco_bash(function_name, *args, split=False):
    """
    Execute a bash function and return the output.

    Args:
        function_name (str): The name of the bash function to execute.
        *args: The arguments to pass to the bash function.
        split (bool, optional): Whether to split the output by lines. Defaults to False.

    Returns:
        str or list: The output of the bash function. If split is True, returns a list of lines, otherwise returns a single string.

    Raises:
        RuntimeError: If there is an error executing the bash function.
    """
    source_command = "source ~/.bash_functions"
    function_call = f'{function_name} {" ".join(map(str, args))}'
    full_command = f"{source_command} && {function_call}"

    # Invoke the bash shell and execute the command
    process = subprocess.Popen(
        ["bash", "-c", full_command],
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

    while stdout[-1] == "\n":
        stdout = stdout[:-1]

    return stdout.split("\n") if split else stdout


def human_time(seconds, dec=2):
    """
    Convert seconds to a human-readable time format.

    Args:
        seconds (int): The number of seconds.
        dec (int, optional): The number of decimal places to include. Defaults to 2.

    Returns:
        str: The human-readable time format.
    """
    s = str(timedelta(seconds=seconds))

    def clean_intra_day(u):
        if len(u.split(":")[0]) == 1:
            u = "0" + u
        if dec == 0 and "." in u:
            u = u.split(".")[0]
        elif dec < 6 and "." in u:
            u = u.split(".")[0] + "." + u.split(".")[1][:dec]
        return u

    units = s.split(", ")
    units[-1] = clean_intra_day(units[-1])
    return ", ".join(units)


def see_fields(obj, *, field, member_paths, idt="    ", level=0):
    """
    Get the values of specified fields in an object.

    Args:
        obj (object): The object to inspect.
        field (str): The name of the field to retrieve.
        member_paths (list): A list of member paths to traverse in the object.
        idt (str, optional): The indentation string. Defaults to "    ".
        level (int, optional): The indentation level. Defaults to 0.

    Returns:
        str: The values of the specified fields in the object.

    Raises:
        ValueError: If member_paths is not a non-empty list of strings, or if a member path does not exist in the object.
    """
    if member_paths is None:
        member_paths = []
    if len(member_paths) == 0:
        raise ValueError("member_paths must be a non-empty list of strings")
    if not isinstance(member_paths, list):
        raise ValueError("member_paths must be a list of strings")
    history = []
    for p in member_paths:
        p = p.split(".")
        c = obj
        s = ""
        for lvl, e in enumerate(p):
            if not hasattr(c, e):
                raise ValueError(
                    f"{c.__class__} does not have member {e}"
                    + f" at level {lvl} of path {p}"
                )
            c = getattr(c, e)
            s += f"{lvl*idt}{e}\n"
            if hasattr(c, field):
                s += f"{(lvl+1)*idt}{field}: {getattr(c, field)}\n"
        history.append(s)
    return "\n".join(history)


def sub_dict(d, keys):
    """
    Create a new dictionary with only the specified keys.

    Args:
        d (dict): The original dictionary.
        keys (list): The keys to include in the new dictionary.

    Returns:
        dict: The new dictionary with only the specified keys.
    """
    return {k: v for k, v in d.items() if k in keys}


def istr(*args, idt_level=0, idt_str="    ", cpl=80):
    """
    Indent and wrap a string.

    Args:
        *args: The strings to be indented and wrapped.
        idt_level (int, optional): The indentation level. Defaults to 0.
        idt_str (str, optional): The indentation string. Defaults to "    ".
        cpl (int, optional): The maximum number of characters per line. Defaults to 80.

    Returns:
        str: The indented and wrapped string.
    """
    wrapper = textwrap.TextWrapper(width=cpl)
    s = "".join(args)
    word_list = wrapper.wrap(text=s)
    base_idt = idt_str * idt_level
    full_idt = base_idt + idt_str
    res = base_idt + word_list[0]
    for line in word_list[1:]:
        res += "\n" + full_idt + line
    return res


def iprint(*args, idt_level=0, idt_str="    ", cpl=80, **kw):
    """
    Print an indented and wrapped string.

    Args:
        *args: The strings to be indented and wrapped.
        idt_level (int, optional): The indentation level. Defaults to 0.
        idt_str (str, optional): The indentation string. Defaults to "    ".
        cpl (int, optional): The maximum number of characters per line. Defaults to 80.
        **kw: Additional keyword arguments to pass to the print function.

    Returns:
        None: The indented and wrapped string is printed.
    """
    print(istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl), **kw)


def iraise(error_type, *args, idt_level=0, idt_str="    ", cpl=80):
    """
    Raise an exception with an indented and wrapped error message.

    Args:
        error_type (type): The type of the exception to raise.
        *args: The strings to be indented and wrapped.
        idt_level (int, optional): The indentation level. Defaults to 0.
        idt_str (str, optional): The indentation string. Defaults to "    ".
        cpl (int, optional): The maximum number of characters per line. Defaults to 80.

    Raises:
        error_type: The raised exception with the indented and wrapped error message.
    """
    raise error_type(istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl))


def ireraise(e, *args, idt_level=0, idt_str="    ", cpl=80, idt_further=True):
    """
    Re-raise an exception with an indented and wrapped error message.

    Args:
        e (Exception): The exception to re-raise.
        *args: The strings to be indented and wrapped.
        idt_level (int, optional): The indentation level. Defaults to 0.
        idt_str (str, optional): The indentation string. Defaults to "    ".
        cpl (int, optional): The maximum number of characters per line. Defaults to 80.
        idt_further (bool, optional): Whether to increase the indentation level for the additional strings. Defaults to True.

    Raises:
        Exception: The re-raised exception with the indented and wrapped error message.
    """
    msg = str(e) + "\n"
    exception_type = type(e)
    full = istr(msg, idt_level=idt_level, idt_str=idt_str, cpl=cpl)
    if idt_further:
        idt_level += 1
    full += (
        "\n"
        + cpl * "*"
        + istr(*args, idt_level=idt_level, idt_str=idt_str, cpl=cpl)
        + cpl * "*"
    )
    raise exception_type(full)


def dupe(base, verbose=True):
    out_file = f'{base}.out'
    err_file = f'{base}.err'

    if verbose:
        print(
            f'Duping stdout, stderr to files below\n\n{out_file}\n{err_file}\n\n'
        )
    sys.stdout = open(out_file, 'w')
    sys.stderr = open(err_file, 'w')
