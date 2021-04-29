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
