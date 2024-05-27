#!/bin/bash

GITHUB_HASH=$1

echo "SSH successful"
echo "Commit: $GITHUB_HASH"
echo "Date: $(date)!"
cd ~/.sandbox/IslandOfMisfitToys

eval "$(conda shell.bash hook)"
conda activate dw_sandbox
cd tests

# last_commit_msg_header=$(git log --format=%s -n 1)
# if [[ "$last_commit_msg_header" == *'['* && "$last_commit_msg_header" == *']'* ]]; then
#     # Extract string inside the first pair of brackets
#     marks_msg=${last_commit_msg_header#*[}
#     marks_msg=${marks_msg%%]*}
#     if [[ -n "$marks_msg" ]]; then
#         # If the extracted string is not empty, replace commas with spaces and prefix with -m
#         the_marks="-m ${marks_msg//,/ }"
#     else
#         the_marks=""
#     fi
# else
#     the_marks=""
# fi
pytest -s

pytest_exit_code=$?

# false_failure_modes=$(ls tests/status)
# if [ $(echo $false_failure_modes | wc -w) -gt 0 ]; then
#     for mode in $false_failure_modes; do
#         echo "FAIL: $mode"
#     done
#     exit 1
# fi

# if [ $pytest_exit_code -ne 0 ]; then
#     echo "FAIL: TRUE FAIL pytest"
#     exit $pytest_exit_code
# fi
exit $pytest_exit_code
