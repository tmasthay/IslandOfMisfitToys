#!/bin/bash

# Define the directories
CURRENT_DIR="$ISL/git-hooks"
HOOKS_DIR="../.git/hooks"

# Create the symbolic link
ln -s -f "$CURRENT_DIR/commit-msg" "$HOOKS_DIR/commit-msg"

# Provide feedback
if [ $? -eq 0 ]; then
    echo "Symbolic link created successfully!"
else
    echo "Failed to create symbolic link."
fi

