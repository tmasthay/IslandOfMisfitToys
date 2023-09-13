#!/bin/bash

if [ -z "$ISL" ]; then
    echo "Error: Need to set ISL environment variable for this script to work"
    exit 1
fi

# Define the directories
CURRENT_DIR="$ISL/git-hooks"
HOOKS_DIR="$ISL/.git/hooks"

# Create the symbolic link
ln -s -f "$CURRENT_DIR/commit-msg" "$HOOKS_DIR/commit-msg"

# Provide feedback
if [ $? -eq 0 ]; then
    echo "Symbolic link created successfully!"
else
    echo "Failed to create symbolic link."
fi

