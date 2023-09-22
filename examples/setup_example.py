import os
import argparse

def create_structure(path):
    # Structure definition
    structure = {
        path: {
            'iomt_output': {
                'figs': {'pwd': []},
                'data': {'pwd': []}
            },
            'deepwave': {
                'figs': {'pwd': []},
                'data': {'pwd': []}
            },
            'compare': {
                'figs': {'pwd': []},
                'data': {'pwd': []}
            },
            'pwd': [
                'metadata.pydict',
                f'{path}_iomt.py',
                f'{path}_deepwave.py',
                f'{path}_deepwave_original.py'
            ]
        }
    }

    def create_recursive(base_path, structure):
        for k, v in structure.items():
            if isinstance(v, dict):
                full_path = os.path.join(base_path, k)
                print(f'Making "{full_path}"... ', end='')
                os.makedirs(full_path, exist_ok=True)
                print('SUCCESS')
                create_recursive(full_path, v)
            elif k == 'pwd' and isinstance(v, list):
                for filename in v:
                    file_path = os.path.join(base_path, filename)
                    print(f'Creating "{file_path}"... ', end='')
                    open(file_path, 'a').close()
                    print('SUCCESS')
            else:
                raise TypeError(f"Unexpected (k,v) = ({k},{v})")

    create_recursive('', structure)  # Start with an empty base path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Setup directory structure for an example.")
    parser.add_argument("path", help="Base path for the directory structure.")
    args = parser.parse_args()
    create_structure(args.path)
