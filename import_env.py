from subprocess import check_output as co
import os

def sco(s,split=True):
    res = co(s,shell=True).decode('utf-8')
    if( split ):
        return res.split('\n')[:-1]
    else:
        return res
    
def get_subfolders(path, omissions=[], depth=1):
    omissions = [path + '/%s'%e for e in omissions]
    u = sco('find %s -type d -depth %d | grep -v "[_.].*"'%(path, depth))
    if( len(omissions) > 0 ):
        u = [e for e in u if e not in omissions]
    return u

def get_submodules(path):
    return sco('find %s -type f -name "*.py" -depth 1 | grep -v "_.*"'%(path))

if( __name__ == "__main__" ):
    u = get_subfolders(os.getcwd(), omissions=['examples'])
    v = get_submodules(os.getcwd())

    print(u)
    print(v)