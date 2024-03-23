#!/bin/bash

GITHUB_HASH=$1

echo "SSH successful"
echo "Commit: $GITHUB_HASH"
echo "Date: $(date)!"
cd ~/.sandbox/IslandOfMisfitToys

eval "$(conda shell.bash hook)"
conda activate dw_sandbox
cd tests
pytest -s

pytest_exit_code=$?

false_failure_modes=$(ls tests/status)
if [ $(echo $false_failure_modes | wc -w) -gt 0 ]; then
    for mode in $false_failure_modes; do
        echo "FAIL: $mode"
    done
    exit 1
fi

if [ $pytest_exit_code -ne 0 ]; then
    echo "FAIL: TRUE FAIL pytest"
    exit $pytest_exit_code
fi
exit $pytest_exit_code
