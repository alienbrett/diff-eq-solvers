'''
This version of PDESolver uses Step DOubling with Local Extrapolation
'''
from numerical_eqs.ode.sdole import SDOLESolve



class SDOLEPDESolver:
    
    def step(self, x, t, dt):
        '''Overwritten by child instantiation.
        This should be able to give the dU (not the derivative, the difference)
        This should return (du, error) a tuple, the result and the error estimation for the step
        '''
        pass
    
    def setup(self):
        '''Called before the solver actually runs
        '''
        pass
    
    def process(self, results):
        '''Give the child the chance to modify the output before we're done
        '''
        return results
    
    
    
    def solve( self, u0, mesh, t0, t1,  boundaries = (None, None), **ode_args, ):
        '''Solve the PDE system, using step doubling and local extrapolation
        Defaults for controls are found in SDOLESolver.time_control_defaults
        '''
    
        # Make our boundaries default to +0 von neumann
        self.boundaries = list(map(
            lambda x: {'type': 'neumann', 'f': lambda t: 0.0} if x is None else x,
            boundaries
        ))
        
        self.mesh = mesh
        self.u0 = u0
        
        # Give the child a chance to pre-compute some stuff
        self.setup()
   
        # Now run our ODE solver
        res = SDOLESolve(
            y0 = u0,
            func = self.step,
            t0 = t0,
            t1 = t1,
            # This is important
            # Our step function will do the full deal on its own
            dont_wrap_func=True,
            **ode_args
        )
    
        # Allow the child class to get a turn after this is done
        return self.process(res)
    
    





