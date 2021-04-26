'''
This version of PDESolver uses Step DOubling with Local Extrapolation
'''

import numpy as np
from tqdm import tqdm

from utils import tridiag_mult, gen_mats
from pde import PDESolver
from numerical_eqs.utils import SimTime



class SDOLESolver(PDESolver):
    time_control_defaults = {
        'tol': 1.0e-4,
        'agrow': 1.25,
        'ashrink': 0.8,
        'dtmin': 1e-7,
        'dtmax': 10
    }
    
    
    
    def solve(
         self, U0, mesh,
        boundaries = (None, None),
        time_controls={}
    ):
        '''Solve the PDE system, using step doubling and local extrapolation
        Time controls can be modified by setting parameters in time_controls dict
        Defaults for time controls are found in SDOLESolver.time_control_defaults
        '''
        
        t0, t1 = self.ts
        
        time_controls = { **self.time_control_defaults, **time_controls }
        # Get our time simulator in order
        simtime = SimTime(
            tol     = time_controls['tol'],
            agrow   = time_controls['agrow'],
            ashrink = time_controls['ashrink'],
            dtmin   = time_controls['dtmin'],
            dtmax   = time_controls['dtmax'],
            tstart  = t0,
            tend    = t1,
        )
        
        
        
        
        
        # Clear out any residual M and S that might be here
        for a in ('_M', '_S'):
            if hasattr(self, a):
                delattr(self, a)
        
        # Get our boundaries in order
        self.boundaries = list(map(
            lambda x: {'type': 'neumann', 'f': lambda t: 0.0} if x is None else x,
            boundaries
        ))

        # Install the stuff needed
        self._mesh = mesh
        res = [U0]
        ts  = [t0]
        # err_steps = []
        # err_nonlins = []

        stepsAccepted = 0
        stepsRejected = 0

        pbar = tqdm(total=t1-t0)
        while ts[-1] < t1:
            
            U = res[-1]
            t = ts[-1]
            
            dU = self.step ( t, U, simtime.dt, )
            error, Ut = self.step (
                t,
                U,
                simtime.dt,
                # Just use the default maximum
                # errorfn = 
            )

            # Should we increment step?
            if simtime.advance(error):
                # Store back our info
                res.append(Ut)
                # Move to the next time
                ts.append(simtime.nextStep())
                # Update our progress bar
                pbar.update(ts[-1] - t)
                
                stepsAccepted += 1
                
            # This step was rejected
            else:
                stepsRejected += 1

        pbar.close()
        res = self.process_sol(res, ts)
        self.solution = res
        return self.solution
        
        
    
    
    def step ( self, t, U, dt, errfn=lambda x: np.linalg.norm(x, ord=np.inf) ):
        '''Calculate the local extrapolated value using the specifics supplied by child class
        '''
        S = self.step_raw( t, U, dt ) + U
        
        # Take two steps
        U1 = self.step_raw( t, U, dt * 0.5 ) + U
        D = self.step_raw( t + dt*0.5, U1, dt * 0.5 ) + U

        # This is our backwards difference
        Ut = 2*D - S
        
        # This is the error
        err = errfn ( Ut )
        
        return err, Ut
    
    
    def step_raw ( self, t, U, dt ):
        '''Take a step and return dU at this point
        '''
        pass