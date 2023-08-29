python update_imports.py
pip uninstall -y IslandOfMisfitToys
pip install .

CURR=$(pwd)
cd
python -W ignore -c "import misfit_toys"
python -W ignore -c "import misfit_toys.acoustic_fwi"
python -W ignore -c "import misfit_toys.elastic_fwi"
python -W ignore -c "import misfit_toys.data"
python -W ignore -c "from misfit_toys.elastic_fwi.driver import main; main()"
cd $CURR 
