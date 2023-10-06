import os
import argparse


def create_structure(path):
    # Structure definition
    structure = {
        path: {
            "iomt": {"figs": None, "data": None},
            "dw": {"figs": None, "data": None},
            "compare": {"figs": None, "data": None},
            "pwd": [
                "metadata.pydict",
                f"{path}_iomt.py",
                f"{path}_deepwave.py",
                f"{path}_deepwave_original.py",
            ],
        }
    }

    def create_recursive(base_path, structure):
        for k, v in structure.items():
            if isinstance(v, dict) or v is None:
                full_path = os.path.join(base_path, k)
                print(f'Making "{full_path}"... ', end="")
                os.makedirs(full_path, exist_ok=True)
                print("SUCCESS")
                if v is None:
                    print(f"    GITKEEP...", end="")
                    open(os.path.join(full_path, ".gitkeep"), "a").close()
                    print("SUCCESS")
                else:
                    create_recursive(full_path, v)
            elif isinstance(v, list):
                for filename in v:
                    file_path = os.path.join(base_path, filename)
                    print(f'Creating "{file_path}"... ', end="")
                    open(file_path, "a").close()
                    print("SUCCESS")
            elif v is None:
                file_path = os.path.join(base_path, k, ".gitkeep")
                print(f'Creating "{file_path}"... ', end="")
                open(file_path, "a").close()
                print("SUCCESS")
            else:
                raise TypeError(f"Unexpected (k,v) = ({k},{v})")

    create_recursive("", structure)  # Start with an empty base path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup directory structure for an example."
    )
    parser.add_argument("path", help="Base path for the directory structure.")
    args = parser.parse_args()
    create_structure(args.path)
