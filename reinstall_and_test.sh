python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip install .

CURR=$(pwd)
cd
python -c "import acoustic_fwi"
python -c "import elastic_fwi"
cd $CURR 
