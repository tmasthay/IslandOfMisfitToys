import inspect

def foo():
    calls = inspect.stack()
    names = [e.function for e in calls]
    for name in names:
        print(name)

def bar():
    foo()

def baz():
    bar()

def doo():
    baz()

def car():
    doo()

def gaz():
    car()

gaz()
