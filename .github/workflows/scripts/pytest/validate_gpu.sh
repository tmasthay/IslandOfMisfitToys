#!/bin/bash

PERSONAL_ACCESS_TOKEN=$1

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
if git status -uno --porcelain | grep '.out$' > /dev/null; then
    echo "Adding .out files to commit..."
    git add tests/**/*.out
else
    echo "WARNING: No .out files found to add."
    exit $pytest_exit_code
fi

# Perform the commit
cp .git/COMMIT_EDITMSG .git/COMMIT_EDITMSG.bak
echo >> .git/COMMIT_EDITMSG
echo "----- SQUASHED AUTO-COMMIT FROM GITHUB ACTIONS -----" >> .git/COMMIT_EDITMSG
echo "AUTO: Adding .out files generated from pytest. [skip ci]" >> .git/COMMIT_EDITMSG

if git commit --amend --no-edit --no-verify; then
    echo "Commit successful...modified commit message below"
    cat .git/COMMIT_EDITMSG
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