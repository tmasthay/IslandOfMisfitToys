#!/bin/bash

ROOT_PATH=$(realpath "misfit_toys")
EXCLUDE_REGEX=("__" "outputs" "multirun" "cfg" "GitHookEm")

# Arrays to store directory information
not_a_subpackage=()
no_version_control_python_files=()
some_version_control_python_files=()

# Function to check if a file is under version control
is_under_version_control() {
    git ls-files --error-unmatch "$1" &>/dev/null
}

rel_path() {
    realpath --relative-to="$ROOT_PATH" "$1"
}

valid_subpackage() {
    local dir="$1"

    # Check if there are any *tracked* .py files in the directory
    #     that match the regex pattern needed
    while IFS= read -r file; do
        if is_under_version_control "$file"; then
            return 0
        fi
    done < <(find "$dir" -maxdepth 1 -name "*.py" | grep -v "__")

    # If we haven't returned yet, check for any directories that
    #     are subpackages
    # Regex is global EXCLUDE_REGEX
    while IFS= read -r subdir; do
        # recurse to check if it is a valid subpackage
        if valid_subpackage "$subdir"; then
            return 0
        fi
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d | grep -vE "$(
        IFS=\|
        echo "${EXCLUDE_REGEX[*]}"
    )")

    return 1
}

generate_import_statements() {
    local path=${1:-.}
    local modules=()

    # Find subpackages and generate import statements for them
    while IFS= read -r dir; do
        if valid_subpackage "$dir"; then
            subpackage_name=$(basename "$dir")
            # echo "from . import $subpackage_name"
            modules+=("$subpackage_name")
        fi
    done < <(find $path -mindepth 1 -maxdepth 1 -type d)

    # Find .py files and filter out those containing '__'
    while IFS= read -r file; do
        if is_under_version_control "$file"; then
            module_name=$(basename "$file" .py)
            modules+=("$module_name")
        fi
    done < <(find $path -maxdepth 1 -name "*.py" | grep -v "__")

    # Generate import statements
    for module_name in "${modules[@]}"; do
        echo "from . import $module_name"
    done

    # Generate __all__ list
    echo -n "__all__ = ["
    local first=true
    for module_name in "${modules[@]}"; do
        if [ "$first" = true ]; then
            echo -n "'$module_name'"
            first=false
        else
            echo -n ", '$module_name'"
        fi
    done
    echo "]"
}

process_directory() {
    local dir="$1"
    local python_files=()
    local version_controlled_files=()
    local no_python_no_

    # Find .py files in the directory
    while IFS= read -r file; do
        # Ignore __init__.py files
        if [[ $(basename "$file") != "__"*".py" ]]; then
            python_files+=("$file")
            if is_under_version_control "$file"; then
                version_controlled_files+=("$file")
            fi
        fi
    done < <(find "$dir" -maxdepth 1 -name "*.py")

    # Classify the directory based on the presence of Python files and version control status
    if ! valid_subpackage "$dir"; then
        not_a_subpackage+=("$dir")
    elif [ ${#version_controlled_files[@]} -lt ${#python_files[@]} ]; then
        some_version_control_python_files+=("$dir")
        update_init_py "$dir"
        echo "    PARTIAL: $(rel_path $dir)"
    else
        update_init_py "$dir"
        echo "    $(rel_path $dir)"
    fi
}

update_init_py() {
    local dir="$1"
    local init_file="$dir/__init__.py"
    local temp_file=$(mktemp)

    # Check if __init__.py exists and has a complete docstring
    if [[ -f "$init_file" ]] && grep -q '^"""' "$init_file"; then
        # Extract the existing docstring
        awk '
        BEGIN { found = 0 }
        /^"""/ {
            if (found == 1) { print; exit }
            found++
        }
        { if (found > 0) print }
        ' "$init_file" >"$temp_file"
    else
        # Create a default docstring
        echo '"""' >"$temp_file"
        echo 'TODO: make package-level docstring' >>"$temp_file"
        echo '"""' >>"$temp_file"
    fi

    # Append the generated import statements to the temporary file
    generate_import_statements "$dir" >>"$temp_file"

    # Move the temporary file to the __init__.py
    mv "$temp_file" "$init_file"
}

# Main script
echo "Directories with auto-generated __init__.py files"
find "$ROOT_PATH" -type d | grep -vE "$(
    IFS=\|
    echo "${EXCLUDE_REGEX[*]}"
)" | while IFS= read -r dir; do
    if [ "$dir" != "$ROOT_PATH" ]; then
        process_directory "$dir"
    fi
done

# buggy exception -- deal with later if you want, not totally necessary
process_directory $ROOT_PATH

# Print the summary of directory classifications
if [ ${#no_python_files[@]} -gt 0 ]; then
    echo ""
    echo "DIRECTORIES WITH NO PYTHON FILES:"
    for dir in "${no_python_files[@]}"; do
        echo "    $(rel_path $dir)"
    done
fi

if [ ${#no_version_control_python_files[@]} -gt 0 ]; then
    echo ""
    echo "DIRECTORIES WITH NO PYTHON FILES UNDER VERSION CONTROL:"
    for dir in "${no_version_control_python_files[@]}"; do
        echo "    $(rel_path $dir)"
        for file in "$dir"/*.py; do
            if ! is_under_version_control "$file"; then
                echo "        $(basename $file)"
            fi
        done
    done
fi

if [ ${#some_version_control_python_files[@]} -gt 0 ]; then
    echo ""
    echo "DIRECTORIES WITH SOME PYTHON FILES UNDER VERSION CONTROL BUT NOT ALL:"
    for dir in "${some_version_control_python_files[@]}"; do
        echo "    $(rel_path $dir)"
        for file in "$dir"/*.py; do
            if ! is_under_version_control "$file"; then
                echo "        $(basename $file)"
            fi
        done
    done
fi
