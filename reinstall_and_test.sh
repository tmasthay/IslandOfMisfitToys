python update_imports.py
pip uninstall -y IslandOfMisfitToys
python setup.py sdist bdist_wheel
pip install dist/IslandOfMisfitToys-0.2.1-py2.py3-none-any.whl

CURR=$(pwd)
cd
python -c "import acoustic_fwi"
python -c "import elastic_fwi"
cd $CURR 
