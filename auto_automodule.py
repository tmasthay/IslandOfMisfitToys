import argparse
import os


def generate_automodule_rst_files(dir_path, exclude_dirs):
    """
    Recursively searches for Python modules and generates automodule .rst files for them.
    """
    for root, dirs, files in os.walk(dir_path, topdown=True):
        if( root.startswith('__') ): continue
        dirs[:] = [
            d for d in dirs \
                if d not in exclude_dirs \
                    and not d.startswith(".")
        ]
        for file in files:
            if file.endswith((".py", ".pyx")):
                mod_path = os.path.relpath(os.path.join(root, file), dir_path)
                mod_name = os.path.splitext(file)[0]
                mod_dir = os.path.dirname(mod_path)
                mod_rst_name = f"{mod_name}.rst"
                mod_rst_path = os.path.join(dir_path, "docs", "source", mod_dir, mod_rst_name)
                os.makedirs(os.path.dirname(mod_rst_path), exist_ok=True)
                with open(mod_rst_path, "w") as f:
                    f.write(f".. automodule:: {os.path.splitext(mod_path)[0]}\n")
                    f.write("   :members:\n")

    # update index.rst with toctree
    update_toc_tree(dir_path)


def update_toc_tree(dir_path):
    """
    Updates the toctree in index.rst to include all .rst files in the project.
    """
    index_file = os.path.join(dir_path, "docs", "source", "index.rst")
    with open(index_file, "r") as f:
        index_content = f.readlines()
    toctree_start = index_content.index(".. toctree::\n")
    toctree_end = index_content.index("\n", toctree_start)
    old_toc = index_content[toctree_start + 2:toctree_end]

    new_toc = []
    for root, dirs, files in os.walk(os.path.join(dir_path, "docs", "source")):
        rel_path = os.path.relpath(root, os.path.join(dir_path, "docs", "source"))
        if( rel_path.startswith(".") and rel_path != "." ):
            continue
        for file in files:
            if file.endswith(".rst") and file != "index.rst":
                rst_path = os.path.join(rel_path, file)
                if rst_path in old_toc:
                    continue
                new_toc.append(f"   {rst_path}\n")

    with open(index_file, "w") as f:
        f.writelines(index_content[:toctree_start + 2])
        f.writelines(new_toc)
        f.write("\n")
        f.writelines(index_content[toctree_end:])

def delete_rst_files(dir_path, protect_files):
    """
    Deletes all .rst files in the docs/source directory, except for index.rst and protected files.
    """
    for root, dirs, files in os.walk(os.path.join(dir_path, "docs", "source")):
        rel_path = os.path.relpath(root, os.path.join(dir_path, "docs", "source"))
        if( rel_path != '.' and rel_path.startswith(".") ):
            continue
        for file in files:
            if file.endswith(".rst") and file not in protect_files:
                os.remove(os.path.join(root, file))

def delete_html_files(dir_path, protect_files):
    """
    Deletes all .html files in the docs/build directory, except for index.html and protected files.
    """
    for root, dirs, files in os.walk(os.path.join(dir_path, "docs", "build")):
        rel_path = os.path.relpath(root, os.path.join(dir_path, "docs", "build"))
        if rel_path.startswith("."):
            continue
        for file in files:
            if file.endswith(".html") and file not in protect_files:
                os.remove(os.path.join(root, file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=os.getcwd(), help="directory to search for Python modules")
    parser.add_argument("-e", "--exclude", nargs="*", default=[], help="directories to exclude")
    parser.add_argument("--reset", action="store_true", help="delete all .rst and .html files except for index.rst and index.html")
    parser.add_argument("--reset_protect", nargs="*", default=[], help="list of protected files that won't be deleted")
    parser.add_argument("--run", action="store_true", help="run 'make html' to generate html files after resetting")
    args = parser.parse_args()

    if args.reset:
        protect_files = args.reset_protect + \
            [   
                "index.rst", 
                "index.html",
                "genindex.html",
                "search.html"
            ]
        delete_rst_files(args.directory, protect_files)
        delete_html_files(args.directory, protect_files)
        update_toc_tree(args.directory)
        print("All .rst and .html files (except for protected files) have been deleted.")
    else:
        exclude = ["__pycache__"] + args.exclude
        generate_automodule_rst_files(args.directory, exclude)

    if args.run:
        os.chdir(os.path.join(args.directory, "docs"))
        os.system("make html")
        print("HTML files have been regenerated.")

if __name__ == "__main__":
    main()

