import numpy as np
import tqdm
from numerical_eqs.utils import SimTime


__all__ = ['ODESolve']


def step ( y, t, vf, dt, errorfn ):
	'''Roughly estimate the future position and error of calculation
	given position, callable derivative & jacobian function
	'''
	# Get derivative, given current position
	deriv, jacobian = vf(t,y)

	# Course estimate
	s = y + deriv*dt

	# Middle estimate
	ymid = y + deriv*(dt/2.0)

	# Derivative estimate at mid-point
	derivMid = deriv + jacobian@(ymid-y)

	# Fine estimate
	d = ymid + derivMid*(dt/2.0)

	# Final new estimate
	ynew = 2*d - s

	# Error estimates
	err_step = errorfn(d-s)

	err_nonlin = errorfn( jacobian@(ynew-y) )

	err = err_step + err_nonlin

	return (ynew, err, err_step, err_nonlin)



def SolveStiffODE (
	y0,
	dyf,
	tstart,
	tend,
	dtmin,
	dtmax,
	norm = np.linalg.norm,
	agrow = 1.25,
	ashrink = 0.8,
	tol = 1e-6,
	):
	'''Finds the position of differential system.
	Needs the position at time tstart,
	and derivative function dyf,
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

	ys  = [y0]
	ts  = []
	err_steps = []
	err_nonlins = []

	stepsAccepted = 0
	stepsRejected = 0

	t = tstart
	pbar = tqdm.tqdm(total=tend-tstart)
	while t < tend:
		yt, error, err_step, err_nonlin = step (
			ys[-1],
			t,
			dyf,
			s.dt,
			errorfn = norm
		)

		# Should we increment step?
		if s.advance(error):
			ys.append(yt)
			
			if len(ts) > 1:
				pbar.update(t - ts[-1])
			else:
				pbar.update(t - tstart)
			
			ts.append(t)
			err_steps.append(err_step)
			err_nonlins.append(err_nonlin)

			t = s.nextStep()
			stepsAccepted += 1
		else:
			stepsRejected += 1

	ts.append(t)
	pbar.close()


	ts = np.asarray(ts)
	ys = np.asarray(ys)
	err_steps = np.asarray(err_steps)
	err_nonlins = np.asarray(err_nonlins)
	dts = ts[1:] - ts[:-1]

	return ys, ts, dts, err_steps, err_nonlins, stepsAccepted, stepsRejected
