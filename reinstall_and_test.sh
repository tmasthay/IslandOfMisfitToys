#!/bin/bash

source ~/.bashrc

conda activate dw
echo "CONDA_PREFIX: $CONDA_PREFIX"

CURR=$(pwd)

# Default value for the parameter
RUN_PYTHON_COMMANDS=${1:-0}
REDOWNLOAD_DATA=${2:-0}

python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip uninstall -y masthay_helpers
pip install .

if [[ ! -z "$CONDA_PREFIX" && $REDOWNLOAD_DATA -ne 0 ]]; then
    rm -rf $CONDA_PREFIX/data
    python -W ignore -c "from misfit_toys.examples.download_data import main; main()"
fi

cd

if [[ $RUN_PYTHON_COMMANDS -ne 0 ]]; then
    #python -W ignore -c "from misfit_toys.fwi.driver import main; main()"
    python -W ignore -c "from misfit_toys.fwi.dist_parallel import main; main()"
fi

cd $CURR

