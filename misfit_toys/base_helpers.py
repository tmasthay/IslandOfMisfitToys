from subprocess import check_output as co
from subprocess import CalledProcessError

def sco(s, split=True):
    try:
        u = co(s, shell=True).decode('utf-8')
        if( split ): return u.split('\n')[:-1]
        else: return u
    except CalledProcessError:
        return None



    
