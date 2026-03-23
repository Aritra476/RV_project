# imports
import numpy as np
import matplotlib.pyplot as plt

#constants
c = 299792458  # speed of light in m/s

# wavelength grid space
wavelength = np.linspace(5008.5,5010.5,2000)
# converting to log space
log_lambda = np.log(wavelength)

lambda0 = 5009.5 #center
sigma = 0.01 #width
depth = 0.6
#log space parameters
x0 = np.log(lambda0)
sigma_log = sigma / lambda0

# Gaussian function in log space
line = 1 - depth * np.exp(- 0.5 * ((log_lambda - x0) / sigma_log) ** 2)

#velocities
v = 2e4 #m/s

#shifted velocity
x_shifted = x0 + v/c 

#shifted line
line_shifted = 1 - depth * np.exp(- 0.5 * ((log_lambda - x_shifted) / sigma_log) ** 2)

#multiple lines
line_center = [5009.45,5009.40,5009.50,5009.55,5009.60]
line_depths = [0.6,0.5,0.7,0.9,0.8]
sigmas = [0.01,0.008,0.012,0.015,0.011]

v_true = 2e4 #m/s

# observed spectrum 

observed_clean = np.ones_like(log_lambda)
for ic,d,s in zip(line_center,line_depths,sigmas):
    x0_i = np.log(ic)
    sigma_log_i = s / ic
    x_shifted_i = x0_i + v_true/c

    line_i = 1 - d * np.exp(- 0.5 * ((log_lambda - x_shifted_i) / sigma_log_i) ** 2)
    observed_clean *= line_i

# Monte Carlo setup
noise_level = 0.01  
n_trials = 1  

rv_results = [] 

#velocity grid   
v_grid = np.linspace(-4e4,4e4,5000)  

for i_trial in range(n_trials): 

    # add noise
    noise = np.random.normal(0, noise_level, size=observed_clean.shape)  
    observed_noisy = observed_clean + noise  

    ccf_values = []  

    for v in v_grid:  

        # shift template  
        shifted_template = np.ones_like(log_lambda)  

        for ic,d,s in zip(line_center,line_depths,sigmas):  
            x0_i = np.log(ic)  
            sigma_log_i = s / ic  
            x_shifted_i = x0_i + v/c  

            line_i = 1 - d * np.exp(- 0.5 * ((log_lambda - x_shifted_i) / sigma_log_i) ** 2)  
            shifted_template *= line_i  

        # use noisy observed spectrum  
        observed_norm = observed_noisy - np.mean(observed_noisy)  
        template_norm = shifted_template - np.mean(shifted_template)  

        num = np.sum(observed_norm * template_norm)  
        denom = np.sqrt(np.sum(observed_norm**2) * np.sum(template_norm**2))  

        ccf_value = num / denom  
        ccf_values.append(ccf_value)  

    # peak detection (parabolic refinement)
    i = np.argmax(ccf_values)  

    if i == 0 or i == len(ccf_values) - 1:  
        v_measured = v_grid[i]  
    else:  
        v1, v2, v3 = v_grid[i-1], v_grid[i], v_grid[i+1]  
        c1, c2, c3 = ccf_values[i-1], ccf_values[i], ccf_values[i+1]  

        dv = v2 - v1  
        v_measured = v2 + (c1 - c3)/(2*(c1 - 2*c2 + c3)) * dv  

    rv_results.append(v_measured)  

    #print each iteration  
    print(f"Trial {i_trial+1:02d}: Measured RV = {v_measured:10.3f}")  


# FINAL RESULTS
rv_results = np.array(rv_results)

print("\nFINAL RESULTS")
print(f"Mean RV: {np.mean(rv_results):.3f}")
print(f"RV Std (error): {np.std(rv_results):.3f}")


# plot last CCF
plt.plot(v_grid, ccf_values)  
plt.xlabel("Velocity (m/s)")  
plt.ylabel("CCF")  
plt.title("Cross-Correlation Function")  
plt.show()