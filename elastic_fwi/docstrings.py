import argparse
import importlib.util
import inspect
import typing

def get_class_info(cls):
    class_info = {
        "name": cls.__name__,
        "docstring": inspect.getdoc(cls),
        "attributes": [],
        "methods": {},
        "base classes": [base.__name__ for base in cls.__bases__],
    }
    
    class_annotations = getattr(cls, '__annotations__', {})
    for k, v in class_annotations.items():
        type_suggestion = str(v.__origin__) \
            .replace('class ', '') \
            .replace('<', '') \
            .replace('>', '') \
            .replace("'", '')
        class_annotations[k] = [type_suggestion] + list(v.__metadata__)
    
    # extract info from class' __dict__
    for name, attr in cls.__dict__.items():
        if callable(attr):  
            class_info["methods"][name] = {"args": \
                list(inspect.signature(getattr(cls, name)).parameters.keys())
            }
        else:
            if name.startswith('__'): continue
            if name in class_annotations.keys():
                class_info["attributes"].append((name, class_annotations[name]))
            else:
                class_info["attributes"].append((name, "Not annotated"))

    return class_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="python file to inspect")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("module.name", args.file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class_members = inspect.getmembers(mod, inspect.isclass)

    data = {}
    for name, cls in class_members:
        if cls.__module__ == mod.__name__:
            data[name] = get_class_info(cls)

    print(data)

if __name__ == "__main__":
    main()
