import numpy as np
from numerical_eqs.utils import SimTime






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
	

    
    

    
    
    
    
def SDOLESolve(
	y0,
	func,
    t1,
    t0=0,
    time_controls = {
        'tol': 1.0e-3,
        'agrow': 1.25,
        'ashrink': 0.8,
        'dtmin': 1e-6,
        'dtmax': 100
    },
    explicit_times = {
        'time points' : [],
        'callback': lambda yt, t: yt
    },
    progress=True,
    pbar = None,
    include_t0=True,
):
        '''Solve the PDE system, using step doubling and local extrapolation
        
        If you want to run some function on the current state at specific points in time, use the explicit_times to control this operation. The callback result will be used to replace yt at each time specified
        
        Args:
            y0: Initial value
            func: callable which should evaluate to the derivative of y(t, yt)
            t1: time value upon which to terminate
            t0: start time, corresponding to y0 [default 0]
            time_controls:
                dict, controls the time stepping and error tolerances. See utils.SimTime for more info on these.
                Defaults to {
                    'tol': 1.0e-3,
                    'agrow': 1.25,
                    'ashrink': 0.8,
                    'dtmin': 1e-6,
                    'dtmax': 100
                }
            explicit_times:
                dict, used to evaluate a separate callable which can optionally modify the ODE value at certain times. dict should look like
                {
                    'time points' : [ (t0+t1)/2, ... ],
                    'callback': (lambda yt, t: yt)
                }
                with 'time points' a list of points to evaluate callable on, and 'callback', a function which should return the new value of yt
            progress: boolean, whether to load with TQDM bar
            include_t0: Whether to add y0 and t0 to the lists of results. Defaults to true
            pbar: not for public use, leave this as None unless you know what you're doing
        '''
        
        # Handle progress and possibly load in TQDM
        if progress:
            try:
                from tqdm import tqdm
            except:
                raise ImportError('TQDM module not found. Try installing tqdm or use progress=False to disable the progress bar')
        
        
        
        # Get our time controls in order
        # This makes sure our function can handle either the recursive case or the base level
        time_controls = {
            **{ 'start_dt': time_controls['dtmin'] },
            **{
                'tol': 1.0e-3,
                'agrow': 1.25,
                'ashrink': 0.8,
                'dtmin': 1e-6,
                'dtmax': 100
            },
            **time_controls,
        }
        # time_controls['start_dt'] = time_controls.get('start_dt', time_controls['dtmin'])
        
        
        
        
        should_close_pbar = False
        # Generate our progress bar, possibly
        if progress and (pbar is None):
            pbar = tqdm(total=t1-t0)
            should_close_pbar = True
            
            
        # We might have to aggregate this
        known_times = sorted(set([t0, t1] + np.asarray(explicit_times.get('time points',[])).tolist()))
        # print(known_times)
        
        sol = None
        
        # See exactly what we should be calling
        callback = explicit_times.get( 'callback', (lambda ys, t: ys) )
        
        new_start_dt = None
        if len(known_times) > 2:
            
            res = None
            for i in range(1, len(known_times)):
                
                tt0 = known_times[i-1]
                tt1 = known_times[i]
                y0 = y0 if res is None else res['ys'][-1,:]
                
                # Recursively get the subinterval we're looking for
                new_res = SDOLESolve(
                    y0 = y0,
                    func = func,
                    t1 = tt1,
                    t0 = tt0,
                    time_controls = {**time_controls, 'start_dt': new_start_dt},
                    explicit_times = {},
                    progress=progress,
                    pbar = pbar,
                    include_t0 = ( include_t0 if i == 1 else False )
                )
                new_start_dt = new_res['rolling dt']
                # Combine
                if res is None:
                    res = new_res
                else:
                    for k in ('steps accepted', 'steps rejected'):
                        res[k] += new_res[k]
                        
                    for k in ('ys', 'time'):
                        res[k] = np.concatenate(
                            [
                                res[k],
                                new_res[k],
                            ],
                            axis = 0
                        )
                
                # Explicitly replace the value requested
                res['ys'][-1,:] = callback(
                    res['ys'][-1,:], # yt
                    res['time'][-1], # t
                )
                
            # Go to the end, and this will be returned
            sol = res
                
        # We should actually compute the thing
        # This is the base-level recursive step
        else:
            # Get all our ducks in a row before this thingy
            res = [y0]
            ts  = [t0]

            stepsAccepted = 0
            stepsRejected = 0

            # Get our time simulator in order
            simtime = SimTime(
                tol      = time_controls['tol'],
                agrow    = time_controls['agrow'],
                ashrink  = time_controls['ashrink'],
                dtmin    = time_controls['dtmin'],
                dtmax    = time_controls['dtmax'],
                tstart   = t0,
                tend     = t1,
                start_dt = time_controls['start_dt']
            )


            while ts[-1] < t1:

                yt = res[-1]
                t = ts[-1]

                yt, error = step (
                    x = yt,
                    t = t,
                    vf = func,
                    dt = simtime.dt,
                )

                # Should we increment step?
                if simtime.advance(error):
                    # Store back our info
                    res.append(yt)
                    # Move to the next time
                    ts.append(simtime.nextStep())

                    if progress:
                        # Update our progress bar
                        pbar.update(ts[-1] - t)

                    stepsAccepted += 1

                # This step was rejected
                else:
                    stepsRejected += 1

                    

            sol = {
                'ys': np.asarray(res),
                'time': np.asarray(ts),
                'steps accepted': stepsAccepted,
                'steps rejected': stepsRejected,
                'rolling dt': simtime.ewma,
            }
                
            # Possibly pop off the first value
            if not include_t0:
                sol['ys'] = sol['ys'][1:, :]
                sol['time'] = sol['time'][1:]
                
            
            
        if progress and should_close_pbar:
            pbar.close()
        return sol
