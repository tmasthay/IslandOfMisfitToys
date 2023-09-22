#!/bin/bash

COMMIT_MSG_FILE="$1"

# Define directives as an array
# extraneous comment
DIRECTIVES=(
    "BUG"
    "BUGFIX"
    "FEATURE"
    "DEBUG"
    "CLEAN"
    "DOCS"
    "SAFETY"
    "REFACTOR"
    "REFACTOR_CKPT"
    "TEST"
    "CONFIG"
    "REVERT"
    "DEPRECATE"
    "UPDATE"
    "PERFORMANCE"
    "PERF"
    "DROP"
    "REQUEST"
    "PAUSE_BUG"
    "PAUSE_DOCS"
    "PAUSE_FEATURE"
    "PAUSE_PERF"
    "PAUSE_PERFORMANCE"
    "PAUSE_OVERKILL"
    "PAUSE_DESIGN"
)

# Join array elements into a string pattern
PATTERN="^($(IFS="|"; echo "${DIRECTIVES[*]}")): .+"

# Check the commit message for patterns
# Check the commit message for patterns
if ! grep -qE "$PATTERN" "$COMMIT_MSG_FILE"; then
    echo "Error: Commit message does not follow the required format."
    echo "Please start your commit message with one of the following directives followed by a colon and a description:"
    IFS=$'\n' # Change IFS to newline for this operation
    printf '    %s\n' "${DIRECTIVES[@]}" 
    echo "Example:"
    echo "    FEATURE: Add new feature"
    echo "***COMMIT FAILED: SEE USAGE ABOVE***"
    exit 1
fi

