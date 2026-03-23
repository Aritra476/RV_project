# import lib
import numpy as np
import matplotlib.pyplot as plt
c = 299792.458e8 # speed of light in m/s

v_grid = np.linspace(-5000000, 5000000, 100) # range of velcoity grid from -5,000,000 m/s to 5,000,000 m/s
# creating wavelength  
lambda0 = 5009.5 # rest wavelength in Angstroms
wavelength = np.linspace(5009.45,5009.55,1000000)
sigma = 0.005  # standard deviation
depth = 0.9  # line depth of the absorption line

template = 1 - depth * np.exp(-0.5 * ((wavelength - lambda0) / sigma) ** 2)

v_true = 2000000 # true velocity in m/s
lambda_shifted = lambda0 * (1 + v_true / c)  # shifted wavelength due to Doppler effect

observed = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_shifted) / sigma) ** 2)

noise = 0.02 * np.random.normal(0,1, len(wavelength))  # adding noise
observed += noise

ccf_values = []

for v in v_grid:
    lambda_trial = lambda0 * (1 + v / c)
    template_shifted = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_trial) / sigma) ** 2)
    ccf = np.sum(template_shifted * observed)
    ccf_values.append(ccf)

v_measured = v_grid[np.argmax(ccf_values)]
print(f"Measured velocity: {v_measured}")
print(f"True velocity: {v_true}")

plt.plot(v_grid, ccf_values)
plt.xlabel('Velocity (m/s)')
plt.ylabel('Cross-Correlation Function')
plt.title('Cross-Correlation Function vs Velocity')
plt.show()
