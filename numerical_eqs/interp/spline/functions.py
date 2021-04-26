import numpy as np
import scipy.interpolate


############ P5 Tridiag solver
class TriDiag:
    def __init__(self, a, d, b, n=None):
        if n is None:
            n = len(d)

        self.n = n
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.d = np.asarray(d)






    def mult(self, x):
        '''Multiplies tridiagonal matrix by some vector of length n
        '''
        x = np.asarray(x)

        return (self.d * x) + \
            np.concatenate([self.a * x[1:],[0]]) + \
            np.concatenate([[0], self.b * x[:-1]])






    def solve(self, y, verbose=True):
        '''Finds solution to equation Ax=y for x
        verbose: print arrays after solving
        All changes to internal arrays will be reset after solving
        '''
        y = np.asarray(y)

        # Create checkpoint, so that internal arrays can be manipulated
        checkPoint = (self.a, self.b, self.d)

        if verbose:
            print("Matrix before solve:")
            for label, array in [('a',self.a), ('d',self.d), ('b',self.b)]:
                print('{} array:'.format(label), array)
                print()

        # Forward solve
        for i in range(self.n):
            y[i] = y[i] / self.d[i]
            if i < self.n-1:
                self.a[i] = self.a[i] / self.d[i]
                self.d[i+1] = self.d[i+1] - self.b[i] * self.a[i]
                y[i+1] = y[i+1] - self.b[i] * y[i]

        # Backward solve
        k = self.n-1
        for j in range(k):
            i = k-j-1
            y[i] = y[i] - self.a[i] * y[i+1]

        if verbose:
            print("Matrix after solve:")
            for label, array in [('a',self.a), ('d',self.d), ('b',self.b)]:
                print('{} array:'.format(label), array)
                print()

        # Restore the checkpoint
        self.a, self.b, self.d = checkPoint

        return y

	
	
	
	
	
############ EC1 Spline solver


def derivSolve(x, y, derivl, derivr):
    '''Solve for point derivatives at internal mesh points:
    1) some set of points
    2) function evaluations at those points
    3) derivatives at left and right endpoints
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    hs = x[1:] - x[:-1]
    
    # Above diagonal
    a = np.concatenate(
        [
            [0],
            -2/hs[1:],
        ],
        axis=0
    )
    
    # True diagonal
    d = np.concatenate(
        [
            [1],
            -4*(1/hs[1:] + 1/hs[:-1]),
            [1],
        ],
        axis=0
    )
    
    # Below diagonal
    b = np.concatenate(
        [
            -2/hs[:-1],
            [0]
        ]
    )
    
    # y_i
    solution = y[1:-1] * ( 1/hs[1:]**2 - 1/hs[:-1]**2)
    # y_i-1
    solution += y[:-2] * (1/hs[:-1]**2)
    # y_i+1
    solution += y[2:] * (-1/hs[1:]**2)
    
    solution *= 6
    
    # Assemble y, for Ax=y
    solution = np.concatenate(
        [
            [derivl],
            solution,
            [derivr],
        ],
        axis=0
    )
    
    derivs = TriDiag(a,d,b).solve(solution, verbose=False)
    return derivs
    

    
    
    
    
    
def cubic_spline(x, y, derivl, derivr):
    '''Creates cubic hermite spline with x,y and left and right derivatives
    Mostly just wrapper for derivSolve
    '''
    derivs = derivSolve(x, y, derivl, derivr)
    return scipy.interpolate.CubicHermiteSpline( x, y, derivs )





def periodic_cubic_spline(x, y):
    
    if not np.isclose( [y[0]], [y[-1]]):
        raise RuntimeError(
            'Function not periodic on [{0},{1}], endpoints don\'t match ({2}, {3})'.format(
                x[0],x[-1],
                y[0],y[-1]
            )
        )
    
    x = np.asarray(x)
    y = np.asarray(y)
    hs = x[1:] - x[:-1]
    
    # solve for zero deriv at endpoint, meets function evals at all interior points
    derivs_base = derivSolve(x, y, 0, 0)
    
    # solve for derivative adjustment at endpoints
    derivs_adjust = derivSolve(x, 0*y, 1, 1)
    
    # re-usable dydx array, used for both dydx_0 and dydx_1
    dydxcoeff = np.array([
        -2/hs[-1],
        -4*(1/hs[0] + 1/hs[-1]),
        -2/hs[0]
    ])
    
    # Coefficients for y[...] at endpoint
    ycoeffs = np.array([
        -6/hs[-1]**2,
        -6*(1/hs[0]**2 - 1/hs[-1]**2),
        6*(1/hs[0]**2)
    ])
    
    
    # This is the last [[g_i``]] formula, applied at endpoints
    # used to solve for alpha in writeup
    alpha = ycoeffs @ [y[-2], y[0], y[1]]
    alpha += dydxcoeff @ [derivs_base[-2], derivs_base[0], derivs_base[1]]
    alpha /= -1* dydxcoeff @ [derivs_adjust[-2], derivs_adjust[0], derivs_adjust[1]]
    
    # The final derivatives used in periodic spline
    derivs = derivs_base + alpha * derivs_adjust
    
    # Spline-ify our results
    return scipy.interpolate.CubicHermiteSpline( x, y, derivs )
