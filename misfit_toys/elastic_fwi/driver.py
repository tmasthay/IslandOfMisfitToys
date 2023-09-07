from .seismic_data import *

def main():
    fwi_solver = marmousi_acoustic()
    fwi_solver.fwi()

if( __name__ == "__main__" ):
    main()