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

def data_attr(obj):
    return [
        attr for attr in dir(obj) if \
        not (attr.startswith('__') and attr.endswith('__')) \
        and not callable(getattr(obj, attr))
    ]

def data_attr(obj, exclude_bools=None):
    if exclude_bools is None: exclude_bools = []
    # Your previously defined function
    tmp = [attr for attr in dir(obj) if not (attr.startswith('__') and attr.endswith('__')) and not callable(getattr(obj, attr))]
    check_exclude = lambda x : any([e(x) for e in exclude_bools])
    tmp = [attr for attr in tmp if not check_exclude(attr)]
    return tmp

def see_fields(
    obj_name, 
    obj, 
    *, 
    level=0, 
    idt='    ', 
    field, 
    exclude_lambda=None,
    exclude_classes=None,
    exclude_strings=None
):
    if exclude_lambda is None: exclude_lambda = []
    if exclude_classes is None: exclude_classes = []
    if exclude_strings is None: exclude_strings = []

    data = data_attr(obj)
    res = ''

    def bld(x, l):
        return f'{l*idt}{x}\n'

    for curr in data:
        curr_obj = getattr(obj, curr)
        if( curr in exclude_strings ): continue
        if( any([type(curr_obj) is e for e in exclude_classes]) ): continue
        if( any([e(curr_obj) for e in exclude_lambda]) ): continue

        if curr == field: 
            res += bld(f'{obj_name}.{field} = {getattr(obj, field)}', level)
        else:
            check = any([type(curr_obj) is e for e in exclude_classes])
            input(f'RECURSE: {curr}=={type(curr_obj)}...{check}...{exclude_classes}')
            res += see_fields(
                curr, 
                curr_obj, 
                level=(level+1), 
                idt=idt, 
                field=field,
                exclude_lambda=exclude_lambda,
                exclude_strings=exclude_strings,
                exclude_classes=exclude_classes
            )
    return res

    
