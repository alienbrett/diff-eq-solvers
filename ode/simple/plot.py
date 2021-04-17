import numpy as np
import matplotlib.pyplot as plt
import math

# Grab my code also
from functions import *




# Also perform the f(t,y) = t**m test
for tol in [0.01, 0.001]:
	xs, ts = ODESolve (
		x0 = np.zeros(4),
		vf = lambda t, x: t**np.arange(4),
		tstart = 0,
		tend = 2,
		dtmin = 0.0001,
		dtmax = 0.01,
		tol = tol,
	)
	print(tol, "\t=>\t", xs[-1,:])
print("On my machine, solver for the power equation specified solution looks invariant to tolerance")




print()




print("Now plotting to plot.png")

tolerances = np.array([0.01, 0.001, 0.0001])

# initial position
p0 = [1,0]
v0 = [0,0.3]

# Step size controls
dtmax = 0.01
dtmin = 0.001

# Misc
system0 = np.array([p0,v0]).flatten()
period = 2*np.pi / (2-(0.3)**2) ** (3/2)

energies = []


# Run everything for a given tolerance
fig, axs = plt.subplots(len(tolerances), 2, figsize=(12,12))
for i, row in enumerate(axs):
	
	# Return the time steps, along with calculated position
	xs, ts = ODESolve (
		x0 = system0,
		vf = orbit,
		tstart = 0,
		tend = 3*period,
		dtmin = dtmin,
		dtmax = dtmax,
		tol = tolerances[i],
	)

	# Sun in middle of plot
	row[0].plot(0,0,'ro')
	row[0].scatter(
		xs[:,0],
		xs[:,1],
		s=0.5
	)

	row[0].set_xlim(-0.3, 1.1)
	row[0].set_ylim(-0.35, 0.35)

	row[0].grid()
	row[0].set_ylabel('Tolerance = {}'.format(tolerances[i]))

	
	
	# Energy plot
	e = energy(xs)
	# Convert energy system in total, into net energy loss
	e = e[0] - e
	# Store the largest absolute energy change, and we'll use this in the plot indeces
	energies.append(max(e))

	# Plot total energy loss
	row[1].grid()
	row[1].fill_between(ts,0,e)


# Just set the energy scale in fair way
energies = np.array(energies)
energies = (
	-0.005,
	1.1*np.max(energies)
)

axs[0,0].set_title('Orbit path, solved via ODE')
axs[0,1].set_title('Net energy loss vs time')

for ax in axs[:,1]:
	ax.set_ylim(energies[0], energies[1])
	

plt.tight_layout()
plt.savefig('plot.png')
print("Saved")

