import numpy as np
import math
import time



def timeit(func):
	def timed(*args, **kwargs):
		t1 = time.time()
		res = func(*args, **kwargs)
		t2 = time.time()
		return (t2-t1, res)
	return timed


def powerIntegrate(p, a, b):
	'''Integrate x^p over (a,b)'''
	p += 1
	return (b**p - a**p)/p



@timeit
def gaussIntegrate(
	# Interval
	a,b,
	# Function
	f,
	# Weight function alpha, where w(x) = x^p,
	p,
	# Number of intervals
	n=1
):
	'''Composite Gauss 1-point rule over n subintervals of (a,b), for callable f(x).
	Uses weight function w(x) = x**p
	'''
	# Divide into subintervals
	intervals = np.linspace(a,b,n+1)
	
	# Bounds of intervals, as lists
	ai = intervals[:-1]
	bi = intervals[1:]
	
	# The evaluated points, according to p
	# This only works explicitly for linear w(x), otherwise
	# We would need lagrange solver or something
	pg = powerIntegrate(p+1, ai, bi) / \
		powerIntegrate(p, ai, bi)
	
	# Compute the weight at each xi
	# Weight is just the integral of w(x) on subinterval
	wg = powerIntegrate(p, ai, bi)

	# Evaluate f(xi) for each xi
	fpg = list(map(f, pg))
	
	# Sum wi*f(xi)
	return wg.T @ fpg




cases = [
	(
		'x',
		[(2, 5),]
	),
	
	(
		'x**0.5',
		[ (-0.5, 4), (-0.5, 40) ],
	),
	(
		'(1-math.sin(x))**2',
		[(-0.5, 5), (-0.5, 50), (-0.5, 100)]
	),
	(
		'(x/math.sin(x/2))**0.5',
		[(-0.5, 2), (-0.5, 10)]
	)
]




if __name__ == '__main__':

	for i, z in enumerate(cases):

		# This is case #i, with function 'name'
		name, other = z
		f = lambda x: eval(name)
		
		print('Case #{0}, f(x) = {1}'.format (i, name))

		# Now run over each of the p's and n's
		for p, n in other:
			
			elapsed, integral = gaussIntegrate(
				0,1,
				f,
				p=p,
				n=n
			)

			print('\t(n={0})\t===>\t{1:.5f} ({2:.3f} ms)'.format(n,integral, elapsed * 1000))

		print()


