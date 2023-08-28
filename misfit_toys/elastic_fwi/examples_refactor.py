import matplotlib.pyplot as plt
from masthay_helpers.typlotlib import *
import torch
import os
import deepwave

from .elastic_class import *
from .seismic_data import marmousi_real

def main():
    fwi = marmousi_real()
    fwi.fwi()

if( __name__ == "__main__" ):
    main()