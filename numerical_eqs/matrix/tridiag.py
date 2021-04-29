import numpy as np




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
            






            
# Generate matrix according to specifications in description
def matGen (n):
    d = np.ones(n)*4 + 0.1 * np.arange(n)
    a = np.ones(n-1) + np.arange(n-1)**2 * 0.01
    b = 0.99 * np.ones(n-1) - 0.03 * np.arange(1,n)
    return d,a,b








if __name__ == '__main__':



    # Test this size matrix
    k = 5

    # Generate matrix accordingly
    # We never actually use the full MxM matrix,
    # we just generate the diagonals
    d,a,b = matGen(k)

    # Our wrapper for matrix
    t = TriDiag(a,d,b)

    # Generate arbitrary x
    x = np.arange(k)

    # Pass X through matrix
    y = t.mult(x)


    # Try to solve for the X that generated Ax=y
    print("SOLVING MATRIX")
    print(''.join(['#']*40))
    xp = t.solve(y, verbose=True)

    print(''.join(['#']*40))
    print('Original X:', x)
    print('Solution X:', xp)
    print('Norm of difference:', np.linalg.norm(x-xp))




    # Try to solve for the X that generated Ax=y
    print("SOLVING MATRIX")
    print(''.join(['#']*40))
    xp = t.solve(y, verbose=True)

    print(''.join(['#']*40))
    print('Original X:', x)
    print('Solution X:', xp)
    print('Norm of difference:', np.linalg.norm(x-xp))
