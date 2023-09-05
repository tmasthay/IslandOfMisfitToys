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



    
