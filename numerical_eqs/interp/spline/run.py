import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from functions import *




##### Plot basic spline ######
x = np.array([0, 0.1, 0.2, 0.5, 0.7, 1.0])

spline = cubic_spline( x, x**3, 0, 3 )

x = np.linspace(0,1,50)
y = x**3 + 0.1

yhat = spline(x)

plt.plot(x, y, label='x^3 + 0.1')
plt.plot(x, yhat, label='Spline')

plt.legend()
plt.gca().set_title('Spline approximation of cubic')
plt.savefig('spline_vs_cubic.png')
plt.close()









##### Plot periodic spline ######
f = lambda x: np.sin(2*np.pi * (x + 0.2))
interval = [0,1]

d = 0.01

# Evaluation mesh
x = np.linspace(0, 1, 10)

# Spline construction
spline = periodic_cubic_spline( x, f(x) )
# Ensure periodic
ps = lambda x: spline( x % (interval[1] - interval[0]) )
# Create this sd_ps thingy
sd_ps = lambda x: d**-2 * (ps(x-d) - 2*ps(x) + ps(x+d))
    

x = np.linspace( 0, 1, 100)

fig, axs = plt.subplots(2, figsize=(7,5))

axs[0].plot( x, ps(x), label='ps')
axs[1].plot( x, sd_ps(x), label='sd_ps')

for ax in axs:
    ax.legend()
axs[0].set_title('Periodic spline PS vs SD_PS')
plt.savefig('sd_vs_ps.png')
plt.close()






####### Plot Error metrics on spline ########

# Our functions
fs = [ np.sin, lambda x: x**0.5,]
f_names = ['sin(x)', 'sqrt(x)']

# Function domains
f_intervals = [[0,np.pi/2.0], [1,4]]

# Function endpoint derivatives
f_derivs = [[1, 0], [0.5, 1]]

# Subintervals for sampling
n_intervals = [4, 8, 16, 32]


fig, axs = plt.subplots(2, figsize=(7,5), sharex=True)


for i, f, derivs, interval in zip(range(len(fs)), fs, f_derivs, f_intervals):
    
    abs_error = []
    
    for n in n_intervals:
        
        # Create mesh
        x_spline = np.linspace(interval[0], interval[1], n)
    
        # Spline construction
        spline = cubic_spline(
            x_spline,
            f(x_spline),
            derivs[0],
            derivs[1]
        )
        
        # Spline error sample
        # We evaluate on 10x as many points as the construction mesh
        x_sample = np.linspace(interval[0], interval[1], 10*n)
        
        abs_error.append(
            # Take the highest value
            np.max(
                # of the absolute error
                np.abs(spline(x_sample) - f(x_sample))
            )
        )
    
    # Plot that error
    axs[i].plot(
        np.log10(n_intervals),
        np.log10(abs_error),
        label=f_names[i],
    )
    # Make sure name is present
    axs[i].legend()
    
    
axs[0].set_title('Log(number of intervals) vs Log(maximum error)')
fig.tight_layout()
plt.savefig('spline_error.png')
