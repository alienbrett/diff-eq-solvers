import numpy as np
import scipy
import scipy.linalg

# Base class from which our solver inherits
from numerical_eqs.pde.sdole import SDOLEPDESolver

from utils import *
# from anelastic_solver import AnelasticSolver


def as_interwoven( M ):
    return M.T.reshape(-1)

def as_stacked( M, bandWidth ):
    return M.reshape(-1, bandWidth).T




class AnelasticSolver(SDOLEPDESolver):
    ########### Boilerplate
    def __init__(self, rho_0, rho_0_p, rho_1, rho_1_p, g, err_fn = lambda x: np.linalg.norm(x, ord=np.inf)):
        
        # Set our constants
        self.rho_0 = rho_0
        self.rho_0_p = rho_0_p
        self.rho_1 = rho_1
        self.rho_1_p = rho_1_p
        self.g = g
        
        # The error function we'll use for step doubling purposes
        self.errfn = err_fn
        
    def setup(self, ):
        self.dx = (self.mesh[1:] - self.mesh[:-1])[0]
        
        # Value for p0 () must be imputed (self.u0[2] must be set implicitly)
        v0, v1, p0, p1, h = (self.u0[i] for i in range(self.u0.shape[0]))

        self.u0[2] = p1 + self.g * ( h * self.rho_0 + (1-h) * self.rho_1)
        # Make sure to mask out the zeros that shuld be there
        self.u0[-3:,-1] = 0
        
        # print('u0')
        # print(self.u0)
        
        
    def process(self, results):
        
        results['ys'] = np.asarray(results['ys'])
        results['time'] = np.asarray(results['time'])
        return results
    
    #################################
    def resid(self, U, dU, dt, dx):
        W = U + dU
        # This fixed so much
        R = W.copy()
        
        # Pull out stacked terms
        V0 = W[0]
        V1 = W[1]
        P0 = W[2,:-1]
        P1 = W[3,:-1]
        H  = W[4,:-1]
        
        d_V0 = dU[0]
        d_V1 = dU[1]
        d_P0 = dU[2,:-1]
        d_P1 = dU[3,:-1]
        d_H  = dU[4,:-1]


        ######## EQ 1
        R[0,1:-1] = self.rho_0 * d_V0[1:-1] / dt  + (P0[1:] - P0[:-1]) / dx
        
        ######## EQ 2
        R[1,1:-1] = self.rho_1 * d_V1[1:-1] / dt + (P1[1:] - P1[:-1]) / dx
        
        ######## EQ 3
        R[2,:-1] = (self.rho_0_p * H * d_P0 / dt) + \
                (self.rho_0 * d_H / dt) + \
                (self.rho_0 * (V0[1:] - V0[:-1])/dx)
        
        ######## EQ 4
        R[3,:-1] = (self.rho_1_p * (1-H) * d_P1 / dt) - \
                (self.rho_1 * d_H / dt) + \
                (self.rho_1 * (V1[1:] - V1[:-1])/dx)
        
        ######## EQ 5
        R[4,:-1] = P1 - P0 + self.g * (H * self.rho_0 + (1-H) * self.rho_1)

        return R
    
    
    
    def step_raw(self, x, t, dt):
        U = x
        
        # Get our initial guess
        R_b = self.resid(U, 0 * U, dt, self.dx)
        
        R_b = as_interwoven(R_b)
        

        
        def a_func ( du ):
            du = as_stacked(du, U.shape[0])
            r = self.resid( U, du, dt, self.dx)
            return as_interwoven(r)
            
        A = banded_jacobian_approx(
            f = a_func,
            x0 = as_interwoven( 0 * U ),
            n_low = 3,
            n_high = 3,
            eps=1e-8,
        )
        
        dU = scipy.linalg.solve_banded(
            (3,3),
            A,
            -1 * R_b,
        )
        
        return as_stacked( dU, U.shape[0] )



    def step(self, x, t, dt):
        '''Use Backward Euler
        '''
        U = x
        
        U_course = U + self.step_raw(U, t, dt)
        U_fine_mid = U + self.step_raw(U, t, dt/2.0)
        U_fine = U_fine_mid + self.step_raw(U_fine_mid, t+dt/2.0, dt/2.0)
        
        # We're not correcting for nonlinearity here
        U_new = U_fine
        
        U_err = self.errfn(U_fine - U_course)
        
        return U_new, U_err