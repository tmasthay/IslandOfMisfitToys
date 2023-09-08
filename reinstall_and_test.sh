#!/bin/bash

# Default value for the parameter
RUN_PYTHON_COMMANDS=0

# Check if a command line argument is provided
if [[ $# -gt 0 ]]; then
    RUN_PYTHON_COMMANDS=$1
fi

python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip install .

CURR=$(pwd)
cd

if [[ $RUN_PYTHON_COMMANDS -ne 0 ]]; then
    python -W ignore -c "from misfit_toys.elastic_fwi.driver import main; main()"
fi

cd $CURR

