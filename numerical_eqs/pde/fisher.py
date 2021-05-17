import numpy as np
import scipy
import scipy.linalg
from utils import tridiag_mult, gen_mats
from sdole import SDOLEPDESolver



class FisherEQSolver(SDOLEPDESolver):
    
    eps = 1e-8
    
    def step (self, x, t, dt):
        '''Evaluates both the step dU and the nonlinear error associated with this step
        Note: makes 3 primitive dU evaluations
        '''
        U = x
        D1 = self._step_dU_raw( U, t, dt ) + U

        U1 = self._step_dU_raw( U, t, dt*0.5) + U
        D2 = self._step_dU_raw( U1, t+dt*0.5, dt*0.5) + U1

        Ut = 2*D2 - D1
        err = self.errfn( D2 - D1 )

        return Ut, err
    
    
    
    
    def _handle_f (self, U, t, dt):
        '''evaluate F and numerically generate F and tridiagonal DF on the mesh
        '''
        n = U.size
        
        F = np.zeros(n)
        DF = np.zeros((3, n))
        
        umid = (U[:-1] + U[1:])/2
        
        # Evaluation
        fmid = self.f(self.xmid, umid)
        
        # Estimate derivative numerically
        if isinstance(fmid, tuple) and len(fmid) == 2:
            fmid, dfdu = fmid
        else:
             dfdu = (
                self.f(self.xmid, umid + self.eps) -
                self.f(self.xmid, umid - self.eps)
            ) / (2*self.eps)
        
        fmid = np.broadcast_to(fmid, self.dx.shape)
        dfdu = np.broadcast_to(dfdu, self.dx.shape)

        # Precompute these arrays
        a = fmid * self.dx/2
        b = dfdu * self.dx/4
        
        # Now sum them into the right places
        F[1:]  += a
        F[:-1] += a
        DF[1, :-1] += b
        DF[1, 1:]  += b
        DF[0, 1:]  += b
        DF[2, :-1] += b
    
        # Throw it back    
        return F, DF
        
        
        
    
    
    def _step_dU_raw (self, U, t, dt):
        '''Actually evaluate dU at a given instance
        '''
        F, DF = self._handle_f(U, t, dt)

        Q = self.M / dt + self.theta * (self.S - DF)
        R = F - tridiag_mult(self.S,U)
        
        for i, b in zip((0,-1), self.boundaries):
            # Set the delta numerically, so that dU + U is known (=g(t)) for t+dt
            if b.get('type') == 'dirichlet':
                
                R[i] = b.get('f')(t) - U[i]
                Q[i, 1+3*i] = 0

            # Just add in the flux on the particular side
            elif b.get('type') == 'neumann':
                R[i] += b.get('f')(t)

        
        dU = scipy.linalg.solve_banded(
            (1,1),
            Q, R,
            overwrite_ab = True,
            overwrite_b = True
        )
        
        return dU

    
    
    
    
    def __init__ (
        self, heat_capacity_func, diffusion_func, f_func=lambda x,t : (0*x),
        err_fn = lambda x: np.linalg.norm(x, ord=np.inf),
        theta = 1,
    ):
        self.d = heat_capacity_func 
        self.e = diffusion_func
        self.f = f_func
        self.errfn = err_fn
        self.theta = theta
        
        
    
    def setup ( self ):
        '''Install all the extra bits we'll need to evaluate dU
        '''
        mesh = self.mesh
        
        # Precompute some stuff for speed
        self.dx = mesh[1:] - mesh[:-1]
        self.xmid = (mesh[1:] + mesh[:-1])/2
        
        
        # Generate Matrices
        self.M, self.S = gen_mats(
            # M base
            np.asarray([
                [1/3, 1/6],
                [1/6, 1/3]
            ]),
            # S base
            np.asarray([
                [1,-1],
                [-1, 1]
            ]),
            # Our mesh evaluation points
            mesh,
            # Dimension functions
            self.d,
            self.e,
        )
        
        