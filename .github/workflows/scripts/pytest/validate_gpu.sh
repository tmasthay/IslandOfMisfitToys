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
git add tests/**/*.out
git commit --amend --no-edit --no-verify
# git push --force-with-lease origin HEAD

exit $pytest_exit_code
