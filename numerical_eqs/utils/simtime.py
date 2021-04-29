import numpy as np


class SimTime:
	def __init__(self, tol, agrow, ashrink, dtmin, dtmax, tstart, tend, start_dt=None, alpha=0.1):

		self.tol = tol
		self.agrow = agrow
		self.ashrink = ashrink
		self.dtmin = dtmin
		self.dtmax = dtmax
		
		# Optionally, allow the caller to explicitly start the simtime on a larger step
		#  In some cases, this will provide better performance
		self.dt = dtmin if start_dt is None else start_dt

		self.stepsSinceReject = 0
		self.tstart = tstart
		self.tend = tend
		self.t = tstart
		
		# EWMA or estimates of prior dt size
		self.alpha = alpha
		self.ewma = None
		self.update_ewma()
		
		
	def update_ewma(self):
		if self.ewma is None:
			self.ewma = self.dt
		else:
			self.ewma = (1.0-self.alpha)*self.ewma + self.alpha*self.dt

			

	def nextStep(self):
		if self.t + self.dt >= self.tend:
			self.t = self.tend
		elif self.t + 2*self.dt >= self.tend:
			self.t = (self.t + self.tend) / 2.0
		else:
			self.t = self.t + self.dt
		# self.t = min(self.t + self.dt, self.tend)
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
			self.update_ewma()
			
			return self.dt == self.dtmin


		elif error > 0.75 * self.tol:
			self.dt = max(self.dtmin, self.dt * self.ashrink)
			self.update_ewma()
			return True


		elif self.stepsSinceReject > 0:
			self.dt = min(self.dtmax, self.dt * self.agrow)
			self.update_ewma()
			self.stepsSinceReject = 0
			return True

		else:
			self.stepsSinceReject += 1
			return True