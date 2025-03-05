#!/bin/bash

# @VS@ _file https://tmasthay.github.io/IslandOfMisfitToys/custom_pages/leaderboard/vp/marmousi/deepwave_example/shots16/case_leaderboard/1/index.html 2

p() {
    PYTHON_PREFIX="import sys; ARGS = sys.argv[1:];"
    program="$1"
    shift
    python -c "$PYTHON_PREFIX $program" "$@"
}

if ! command -v curl &>/dev/null; then
    echo "curl is not installed. Please install curl and try again."
    exit 1
fi

if ! command -v xmllint &>/dev/null; then
    echo "xmllint is not installed. Please install xmllint and try again."
    exit 1
fi

if ! command -v python &>/dev/null; then
    echo "python is not installed. Please install python and try again."
    exit 1
fi

# URL to fetch
# CASE_NAME="vp/marmousi/deepwave_example/shots16/case_leaderboard/1"
# SUFFIX="index.html"
# URL="$PREFIX/$CASE_NAME/$SUFFIX"
DOCS_URL="https://tmasthay.github.io/IslandOfMisfitToys"
REPO_URL="https://github.com/tmasthay/IslandOfMisfitToys"

PREFIX="$DOCS_URL/custom_pages/leaderboard"
URL="$1"
VERBOSE=${2:-0}

TARGET_BRANCH="$(p 'print("_".join(ARGS[0].replace(ARGS[1], "").split("/")[1:-1]))' $URL $PREFIX)"

CURRENT_BRANCH=$(git branch --show-current)

# check if URL begins with same prefix
if [[ ! $URL == $PREFIX* ]]; then
    echo "Fatal: URL=$URL does not begin with $PREFIX"
    exit 1
fi

# Fetch and extract the relevant content
CONTENT=$(curl -s "$URL" | xmllint --html --xpath '//p[@class="admonition-title" and text()="git_info.txt"]/following-sibling::div[1]/div/pre/text()' - 2>/dev/null)

# Extract HASH (first occurrence of "HASH: some_hash")
HASH=$(echo "$CONTENT" | grep -oP '^HASH: \K[0-9a-f]+')

if ! git switch -c $TARGET_BRANCH $HASH; then
    echo "Fatal: Unable to checkout branch $TARGET_BRANCH"
    exit 1
fi

# Generate the separator line using Python
SEPARATOR=$(p 'print(80 * "*")')

# Extract the patch (everything after the separator line)
echo "$CONTENT" | awk -v sep="$SEPARATOR" 'index($0, sep) {flag=1; next} flag' >/tmp/leaderboard_patch.diff

if [ "$VERBOSE" -ge 1 ]; then
    echo "Metadata successfully extracted from $URL"
fi
if [ "$VERBOSE" -ge 2 ]; then
    echo "Extracted Hash: $HASH"
    echo "Extracted Patch:"
    cat /tmp/leaderboard_patch.diff
fi

# apply the patch and push to remote
if ! git apply /tmp/leaderboard_patch.diff; then
    echo "Fatal: Unable to apply patch"
    exit 1
fi
git add -u
if ! git commit --no-verify -m "LEADERBOARD_TMP_BRANCH: $URL"; then
    echo "Fatal: Unable to commit changes"
    exit 1
fi
if ! git push -u origin $TARGET_BRANCH; then
    echo "Fatal: Unable to push changes to $TARGET_BRANCH"
    exit 1
fi

# switch back to the original branch
git switch $CURRENT_BRANCH

# give command for a difftool locally
echo "To view how your current branch differs from this previous test, either "
echo "    git difftool $CURRENT_BRANCH..$TARGET_BRANCH"
echo "    or visit the following URL:"
echo "    $REPO_URL/compare/$CURRENT_BRANCH..$TARGET_BRANCH"

echo to "To see the diff introduced by the patch file, either run "
echo "    git difftool $HASH..$TARGET_BRANCH"
echo "or visit the following URL:"
echo "       $REPO_URL/compare/$HASH..$TARGET_BRANCH"
