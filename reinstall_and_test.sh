#!/bin/bash

source ~/.bashrc

conda activate dw
echo "CONDA_PREFIX: $CONDA_PREFIX"

CURR=$(pwd)

# Default value for the parameter
RUN_PYTHON_COMMANDS=0

# Check if a command line argument is provided
if [[ $# -gt 0 ]]; then
    RUN_PYTHON_COMMANDS=$1
fi

python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip install .


if [ ! -z "$CONDA_PREFIX" ]; then
    rm -rf $CONDA_PREFIX/data
    cd examples; python data_fetching.py; cd ..
fi

cd

if [[ $RUN_PYTHON_COMMANDS -ne 0 ]]; then
    #python -W ignore -c "from misfit_toys.fwi.driver import main; main()"
    python -W ignore -c "from misfit_toys.fwi.dist_parallel import main; main()"
fi

cd $CURR

