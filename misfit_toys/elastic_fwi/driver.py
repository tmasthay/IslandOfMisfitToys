from .seismic_data import *

def main():
    fwi_solver, model, uniform_survey = marmousi_acoustic()
    fwi_solver.fwi(make_plots=['vp'])

if( __name__ == "__main__" ):
    main()
