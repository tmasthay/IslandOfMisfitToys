import subprocess
from subprocess import check_output as co
from subprocess import CalledProcessError
from datetime import timedelta

def sco(s, split=True):
    try:
        u = co(s, shell=True).decode('utf-8')
        if( split ): return u.split('\n')[:-1]
        else: return u
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
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()

    # Decode the output and error bytes to string
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    # If there's an error, raise it
    if stderr:
        raise RuntimeError(f"Error executing '{function_call}': {stderr}")
    
    while( stdout[-1] == '\n' ):
        stdout = stdout[:-1]

    return stdout.split('\n') if split else stdout

def human_time(seconds, lengths=[1, 2, 2, 2]):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = []
    
    if days > 0:
        day_str = f"{days:0{lengths[0]}d}"
        result.append(day_str)
        
    time_str = (f"{hours:0{lengths[1]}d}:{minutes:0{lengths[2]}d}:"
                f"{seconds:0{lengths[3]}d}")
    result.append(time_str)
    
    return ":".join(result) if days > 0 else result[0]

def see_fields(obj, *, field, member_paths, idt='    ', level=0):
    if( member_paths is None ): member_paths = []
    if( len(member_paths) == 0 ):
        raise ValueError('member_paths must be a non-empty list of strings')
    if( not isinstance(member_paths, list) ):
        raise ValueError('member_paths must be a list of strings')
    history = []
    for p in member_paths:
        p = p.split('.')
        c = obj
        s = ''
        for (l,e) in enumerate(p):
            if( not hasattr(c, e) ):
                raise ValueError(
                    f'{c.__class__} does not have member {e}' + 
                    f' at level {l} of path {p}'
                )
            c = getattr(c, e)
            s += f'{l*idt}{e}\n'
            if( hasattr(c, field) ):
                s += f'{(l+1)*idt}{field}: {getattr(c, field)}\n'
        history.append(s)
    return '\n'.join(history)
