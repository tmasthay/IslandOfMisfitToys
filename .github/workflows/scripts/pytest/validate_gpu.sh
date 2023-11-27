#!/bin/bash

echo "SSH successful at $(date)!"
source ~/.bashrc
cd $ISL
conda activate dw
pytest tests
pytest_exit_code=$?

echo "SLEEPING FOR 30 SECONDS"
echo "CANCEL NOW IF YOU NEED TO CANCEL AUTO-COMMIT"
sleep 30

# Check if there are any .out files to add
if git ls-files --others --exclude-standard | grep -q 'tests/.*\.out$'; then
    echo "Adding .out files to commit..."
    git add tests/**/*.out
else
    echo "WARNING: No .out files found to add."
    exit $pytest_exit_code
fi

# Perform the commit
if git commit --amend --no-edit --no-verify; then
    echo "Commit successful."
else
    echo "WARNING: Commit failed. Not pushing changes."
    exit $pytest_exit_code
fi

# Push only if commit is successful
if git push --force-with-lease origin HEAD; then
    echo "Push successful."
else
    echo "WARNING: Push failed."
    exit $pytest_exit_code
fi

exit $pytest_exit_code
