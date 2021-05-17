import numpy as np

def tridiag_mult(M, x):
    '''Multiply tridiagonal matrix by a vector
    '''
    # print(M.shape)
    # print(x.shape)
    
    j = np.multiply( M, x )    
    # print(j)
    
    res = j[1]
    res[:-1] += j[0,1:]
    res[1:] += j[-1,:-1]
    return res



def gen_mats ( m_m, m_s, mesh, d, e):
    '''Generates M, S mass and stiffness matrices,
    according to some specifications
    '''
    n = mesh.size
    
    x_mid = (mesh[1:] + mesh[:-1]) / 2.0
    dx = mesh[1:] - mesh[:-1]
    d_x_mid = np.broadcast_to(d(x_mid), dx.shape)
    e_x_mid = np.broadcast_to(e(x_mid), dx.shape)
    
    M = np.zeros((3,n))
    S = np.zeros((3,n))

    m = np.asarray([
        [0, m_m[0,1]],
        [m_m[0,0],m_m[1,1]],
        [m_m[1,0],0]
    ])
    s = np.asarray([
        [0, m_s[0,1]],
        [m_s[0,0],m_s[1,1]],
        [m_s[1,0],0]
    ])
    
    for i in range(n-1):
        M[:,i:i+2] += m * d_x_mid[i] * dx[i]
        S[:,i:i+2] += s * e_x_mid[i] / dx[i]

    return M, S



def banded_jacobian_approx(f, x0, n_low, n_high, eps=1e-8, *args, **kwargs):
    '''Finite-difference approximation of a function with sparse banded jacobian
    
    callable f, on the numeric approximation, for eps in the i'th position,
        should return nonzero nowhere except in the interval of positions [i-n_low, i+n_high]
    
    Designed to minimize evaluations to the callable
    '''    

    # # Tests for banded jacobian solve
    # k = 10

    # f0 = lambda x: tridiag_mult(
    #     np.asarray([
    #         np.arange(0, k),
    #         np.ones(k),
    #         k-1-np.arange(0, k),
    #     ]),
    #     (x - np.ones(k)),
    # )
    # x00 = np.zeros(k)
    # f1 = lambda x: 4 * (x+1)
    # x01 = np.zeros(6)
    # banded_jacobian_approx(
    #     f = f0,
    #     x0 = x00,
    #     n_low = 1,
    #     n_high = 1,
    #     eps = 0.03
    # )

    # Some sizes
    n_x = x0.size
    k = n_low + n_high + 1
    
    # Get the initial position
    f_x0 = f(x0, *args, **kwargs)
    
    # Our eventual result
    A = np.zeros((k, n_x))

    # This will make computation a tiny bit more readable hopefully?
    xs = np.arange(0,n_x)
    
    for i in range(k):
        # Where our eval points
        s = slice(i, n_x, k)
        # Get our (x0 + eps)
        x1 = x0.copy()
        x1[s] += eps
        # Perform the f eval
        f_x1 = f(x1, *args, **kwargs)
        # The numerical derivative
        d_x1 = (f_x1 - f_x0)/eps
        
        # Take each island and aggregate into A
        for j in xs[s]:
            # Get the indeces of values we want
            raw_index = np.arange(j-n_low, j+n_high+1)
            index = np.mod( raw_index, n_x )
            
            d = np.take_along_axis(d_x1, index, 0)
            
            # Mask out values that aren't in the region we want
            A[:, j] = np.where( raw_index * (n_x - raw_index - 1) < 0, 0, d )
            
    
    return A

