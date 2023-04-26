from dolfin import *
import sys, os
from hippylib import *

class PML:
    def __init__(self, mesh, box, box_pml, A):

        #loop through and define the distance to each side of the domain
        t = [None] * 4
        for i in range(4):
            t[i] = box_pml[i] - box[i]
            if( abs(t[i]) < DOLFIN_EPS ):
                t[i] = 1.0
    
        #define the PML in the X direction
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

        #define the PML in the Y direction
        #TODO: come back and modify this after we get it running
        #   the problem we are looking to solve is different 
        #   when it comes to the Y direction
        #   (just omit the top boundary part and implement zero Neumann there)  
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
 
        #differentiate between physical and artificial domain
        physical_domain = AutoSubDomain(
            lambda x, 
            on_boundary: \
                x[0] >= box[0] \
                and x[0] <= box[2] \
                and x[1] >= box[1] \
                and x[1] <= box[3]
        )

        #mark cells 0 on the PML and 1 inside the physical domain
        cell_marker = MeshFunction('size_t', mesh, mesh.geometry.dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = Measure('dx', subdomain_data=cell_marker)

       
