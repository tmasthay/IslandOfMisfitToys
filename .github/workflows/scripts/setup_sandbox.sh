#!/bin/bash

GITHUB_HASH=$1

cd ~/.sandbox

eval "$(conda shell.bash hook)"


# Ensure we are not in any conda environment
# conda deactivate

# Removing the existing environment (if it exists)
conda env remove --name dw_sandbox --yes || {
    echo "FAIL: Failed to remove the existing conda environment."
    exit 1
}

# Creating a new environment
conda create -n dw_sandbox python=3.10 --yes || {
    echo "FAIL: Failed to create a new conda environment."
    exit 1
}

# Activating the new environment
conda activate dw_sandbox

# Initialize a new Git repository
mkdir IslandOfMisfitToys && cd IslandOfMisfitToys
git init || { echo "FAIL: Failed to initialize a git repository."; exit 1; }

# Add the remote and fetch the specific commit
git remote add origin https://github.com/tmasthay/IslandOfMisfitToys.git
git fetch --depth 1 origin $GITHUB_HASH || {
    echo "FAIL: Failed to fetch the commit $GITHUB_HASH."
    exit 1
}

# Checkout the specific commit
git checkout $GITHUB_HASH || {
    echo "FAIL: Failed to checkout the commit $GITHUB_HASH."
    exit 1
}

echo "Successfully set up the environment and checked out the commit."
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PWD=$PWD"
echo "pip=$(which pip)"
echo "INSTALLING IOMT"
time pip install -e . || {
    echo "FAIL: Failed to install the package."
    exit 3
}
exit 0
