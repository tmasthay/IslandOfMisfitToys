import re
import argparse

def check_commit_msg(commit_msg_file):
    # Define directives as a dictionary
    directives = {
        "BUG": "Found bug",
        "BUGFIX": "Fixed bug",
        "FEATURE": "Added feature",
        "DEBUG": "Debugging change",
        "CLEAN": "Cleaned code",
        "DOCS": "Documentation change",
        "SAFETY": "Safety enhancement",
        "REFACTOR": "Refactored code",
        "REFACTOR_CKPT": "Refactoring checkpoint",
        "TEST": "Test addition/change",
        "CONFIG": "Config change",
        "REVERT": "Reverted change",
        "DEPRECATE": "Deprecated feature/code",
        "UPDATE": "General update",
        "PERFORMANCE": "Performance optimization",
        "PERF": "Performance change",
        "DROP": "Dropped code/feature",
        "REQUEST": "Request-based change",
        "FOLLOW": "Follow-up to previous commit",
        "REMOVE": "Removed code, feature or file",
        "RM": "Removed code, feature or file",
        "PAUSE_BUG": "Pausing due to a bug",
        "PAUSE_DOCS": "Pausing doc changes",
        "PAUSE_FEATURE": "Pausing feature work",
        "PAUSE_PERF": "Pausing performance work",
        "PAUSE_OVERKILL": "Pausing due to overkill",
        "PAUSE_DESIGN": "Pausing for design reasons",
        "BUGREV": "Reverted bugfix; bug found again"
    }

    # Extract directive names
    directive_names = list(directives.keys())

    # Create the regex pattern
    pattern = "^(%s): .+" % "|".join(directive_names)

    # Read the commit message from file
    with open(commit_msg_file, "r") as f:
        commit_msg = f.read()

    # Check if the commit message matches the pattern
    if not re.match(pattern, commit_msg):
        print("Error: Commit message does not follow the required format.")
        print("Please start your commit message with one of the following directives followed by a colon and a description:")

        # Determine the length of the longest directive for alignment
        max_length = max(len(name) for name in directive_names)

        # Print directives and descriptions in a formatted manner
        for name, description in directives.items():
            print(f"    {name.ljust(max_length)}: {description}")

        print("Example:")
        print(f"    {'FEATURE'.ljust(max_length)}: Add new feature")
        print(f"    {'BUG'.ljust(max_length)}: Fix bug in feature")
        print("***COMMIT FAILED: SEE USAGE ABOVE***")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the format of a git commit message.")
    parser.add_argument('commit_msg_file', type=str, help="Path to the commit message file.")

    args = parser.parse_args()
    check_commit_msg(args.commit_msg_file)

