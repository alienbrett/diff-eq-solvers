import numpy as np
import scipy.linalg

from utils import tridiag_mult, gen_mats
from pde import PDESolver



class WavePDESolver(PDESolver):
    
    def __init__( self, t0, t1, d_func, e_func ):
        self.ts = (t0, t1)
        
        self.d_func = d_func
        self.e_func = e_func
    
    
    def combine_dU(self, U, dU):
        '''Generic of U + dU
        This function can be overwritten to allow for U to be tuple or generic
        '''
        u, v = U
        du, dv = dU
        
        return (u+du, v+dv)
    
    
    def process_sol(self, res, t):
        '''Converts the raw loop output into a meaningful solution
        '''
        U = [ u for u,v in res ]
        V = [ v for u,v in res ]
        
        res = {
            'U': np.asarray(U),
            'dU': np.asarray(V),
            'time': t,
            'mesh': self._mesh,
        }
        
        return res

    
    def step (self, t, dt, U, ):
        '''Solve for time derivative of temperature samples that satisfy PDE equation
        Uses theta method,
        Accomadates arbitrary neumann or dirichlet boundaries
        
        Follows the process
        1) Unpack our values and comput the matrices we'll use
        2) Apply neuman conditions to the matrices, where applicable
        3) Solve the tridiagonal and resolve dv
        4) use dv to compute du
        5) Apply dirichlet conditions to the values of du and dv where applicable
        '''
        M, S = self.mats
        
        # Unpack u
        u, v = U

        # Apply neuman boundary
        for i, b in zip((0,-1), self.boundaries):
            # Just add in the flux on the particular side
            if b.get('type') == 'neumann':
                v[i] += b.get('f')(t + dt/2.0)        
            
    
    
        ### First, we solve for dV
        # dV satisfies Q@dV=R
        Q = (M / dt) + (dt/4.0) * S
        R = -1 * tridiag_mult( S, u + 0.5*dt*v )
        
        
        dv = scipy.linalg.solve_banded(
            (1,1),
            Q, R,
            # These give us a minute speed advantage
            overwrite_ab = True,
            overwrite_b = True
        )
        ### Then, dU is easy
        du = dt *(v + 0.5 * dv)
        
        # Apply dirichlet boundary
        for i, b in zip((0,-1), self.boundaries):
            # Just add in the flux on the particular side
            if b.get('type') == 'dirichlet':
                du[i] = b.get('f')(t + dt) - u[i]
                # Store back the correct values to dv, implied from du being changed
                dv = 2*((1.0/dt)*du - v)

        return du, dv