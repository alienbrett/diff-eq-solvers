import numpy as np
import matplotlib.pyplot as plt
import math


def basicStep ( x, t, vf, dt ):
	'''Euler step, but using derivative at middle point of dt
	'''
	dt2 = dt/2.0
	
	tmid = t + dt2
	
	w = x + dt2 * vf(t, x)
	
	return x + dt * vf( tmid, w )



def step ( x, t, vf, dt ):
	'''Roughly estimate the future position and error of calculation
	given position, callable derivative function
	'''
	xcourse = basicStep ( x, t, vf, dt )
	
	xmid = basicStep ( x, t, vf, dt/2.0 )
	xfine = basicStep ( xmid, t+dt/2.0, vf, dt/2.0 )
	
	xnew = xfine * 4/3.0 - xcourse * 1/3.0
	
	# Return our guess, and the estimated error
	return (xnew, xfine - xcourse)
	
	
	
class SimTime:
	def __init__(self, tol, agrow, ashrink, dtmin, dtmax, tstart, tend):
		
		self.tol = tol
		self.agrow = agrow
		self.ashrink = ashrink
		self.dtmin = dtmin
		self.dtmax = dtmax
		self.dt = dtmin
		
		self.stepsSinceReject = 0
		self.tstart = tstart
		self.tend = tend
		self.t = tstart


	def nextStep(self):
		self.t = min(self.t + self.dt, self.tend)
		return self.t
	
	
	def advance (self, error):
		'''
		If error very big,
			dtFine
		if error sorta big,
			dtFiner
		if error not big,
			dtCourser
		'''
		error = np.linalg.norm(error)
	
		if error > self.tol:
			self.dt = max(self.dtmin, self.dt / 2.0 )
			self.stepsSinceReject = 0
			
			return self.dt == self.dtmin
		
		
		elif error > 0.75 * self.tol:
			self.dt = max(self.dtmin, self.dt * self.ashrink)
			return True
			
		
		elif self.stepsSinceReject > 0:
			self.dt = min(self.dtmax, self.dt * self.agrow)
			self.stepsSinceReject = 0
			return True
		
		else:
			self.stepsSinceReject += 1
			return True
	
	
	
def ODESolve (
	x0,
	vf,
	tstart,
	tend,
	dtmin,
	dtmax,
	agrow = 1.25,
	ashrink = 0.8,
	tol = 1e-6,
):
	'''Finds the position of differential system
	Needs the position at time tstart,
	and derivative function vf,
	including some control parameters around the solving algorithm
	'''
	
	s = SimTime(
		tol = tol,
		agrow = agrow,
		ashrink = ashrink,
		dtmin = dtmin,
		dtmax = dtmax,
		tstart = tstart,
		tend = tend,
	)
	
	xs = [x0]
	ts = [tstart]
	
	t = tstart
	while t != tend:
		# Well we need to take at least one step!
		xt, error = step (
			xs[-1],
			ts[-1],
			vf,
			s.dt
		)
		
		# Should we increment step?
		if s.advance(error):
			xs.append(xt)
			ts.append(t)
			t = s.nextStep()
	
	return np.asarray(xs), np.asarray(ts)




def orbit (t, x):
	'''Computes derivative of object, at iven 4-value vector, [pos, vel]'''
	norm = np.linalg.norm(x[:2],2) ** 3
	return np.array(
		[
			x[2],
			x[3],
			-x[0]/ norm,
			-x[1]/ norm,
		]
	)



def energy(xs):
	'''Sum energy of the system at specific time
	Compute's along axis 0, and axis 1 holds position/velocity
	'''
	return 1/2 * np.linalg.norm(xs[:,2:],axis=1)**2 - np.linalg.norm(xs[:,:2],axis=1)**-1
