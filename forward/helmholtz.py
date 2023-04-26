from dolfin import *
import sys, os
from hippylib import *

class PML:
    def __init__(self, mesh, box, box_pml, A):
        t = [None] * 4
        for i in range(4):
            t[i] = box_pml[i] - box[i]
            if( abs(t[i]) < DOLFIN_EPS ):
                t[i] = 1.0
    
        self.sig_x = Expression(
            '(x[0] < xL) * A * (x[0] - xL) * (x[0] - xL) / (tL*tL) + %s'%(
                '(x[0] > xR) * (x[0] - xR) / (tR*tR)'
            ),
            xL=box[0],
            xR=box[2],
            A=A,
            tL=t[0],
            tR=t[2],
            degree=2
        )

        self.sig_y = Expression(
            '(x[1] < yB) * A * (x[1] - yB) * (x[1] - yB) / (tB*tB) + %s'%(
                '(x[1] > yT) * A * (x[1] - yT) * (x[1] - yT) / (tT * tT)'
            ),
            yB=box[1],
            yT=box[3],
            A=A,
            tB=t[1],
            tT=t[3],
            degree=2
        )
 
        physical_domain = AutoSubDomain(
            lambda x, 
            on_boundary: \
                x[0] >= box[0] \
                and x[0] <= box[2] \
                and x[1] >= box[1] \
                and x[1] <= box[3]
        )
       
