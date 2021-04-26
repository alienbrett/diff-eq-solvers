import numpy as np
from tqdm import tqdm

from utils import tridiag_mult, gen_mats




class PDESolver:
    '''Approximates numerically a specific PDE equation
    '''
    M_base = np.asarray([
        [1/3,1/6],
        [1/6,1/3]
    ])
    S_base = np.asarray([
        [1,-1],
        [-1,1]
    ])
    
    
    def __init__( self, t0, t1):
        self.ts = (t0, t1)
        
    @property
    def mats ( self ):
        '''Returns M and S matrices
        '''
        if not ( hasattr(self, '_M') and hasattr(self, '_S') ):
            self._M, self._S = gen_mats(
                self.M_base,
                self.S_base,
                mesh = self._mesh,
                d = self.d_func,
                e = self.e_func
            )
        return self._M, self._S

    
    def step (self, t, dt, U, ):
        '''Solve for time derivative of temperature samples that satisfy PDE equation
        Uses theta method,
        Accomadates arbitrary neumann or dirichlet boundaries,
        '''
        pass
    
    
    def combine_dU(self, U, dU):
        '''Generic of U + dU
        This function can be overwritten to allow for U to be tuple or generic
        '''
        return U + dU
    
    def process_sol(self, res, ts):
        '''Converts the raw loop output into a meaningful solution
        '''
        res = {
            'res': np.asarray(res),
            'time': ts
        }
        
        return res
    
    
    def solve( self, U0, mesh, dt, boundaries = (None, None), progress=True ):
        '''Solve the PDE system
        '''
        
        # Clear out any residual M and S that might be here
        for a in ('_M', '_S'):
            if hasattr(self, a):
                delattr(self, a)
        
        self.boundaries = list(map(
            lambda x: {'type': 'neumann', 'f': lambda t: 0.0} if x is None else x,
            boundaries
        ))

        self._mesh = mesh
        
        res = [U0]
        t0, t1 = self.ts
        ts  = np.arange(t0, t1+dt, dt)

        loop = ts[1:]
        
        # Loop across static time samples
        for t in (pbar if progress else loop):

            # Load current temperature
            U = res[-1]

            # Solve for global derivative that satisfies
            dU = self.step(
                t = t,
                dt = dt,
                U = U,
            )
            
            # Store the new temperature back
            res.append(self.combine_dU(U, dU))

        # Aggregate the data we used into a meaningful summary
        res = self.process_sol(res, ts)
        # Store summary
        self.solution = res
        
        return self.solution
        
    
