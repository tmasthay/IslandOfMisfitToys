#!/bin/bash

COMMIT_MSG_FILE="$1"
DIRECTIVES="BUG|BUGFIX|FEATURE|DEBUG|CLEAN|DOCS|SAFETY|REFACTOR|REFACTOR_CKPT|STYLE|TEST|INIT|CONFIG|REVERT|DEPRECATE|UPDATE|CHORE|PERFORMANCE"
PATTERN="^($DIRECTIVES): .+"

# Check the commit message for patterns
if ! grep -qE "$PATTERN" "$COMMIT_MSG_FILE"; then
    echo "Error: Commit message does not follow the required format."
    echo "Please start your commit message with one of the following directives followed by a colon and a description:"
    echo "$DIRECTIVES"
    exit 1
fi

