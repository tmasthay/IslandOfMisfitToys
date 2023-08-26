python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip install .

CURR=$(pwd)
cd
python -c "import misfit_toys"
python -c "import misfit_toys.acoustic_fwi"
python -c "import misfit_toys.elastic_fwi"
python -c "import misfit_toys.data"
cd $CURR 
