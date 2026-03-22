# import lib
import numpy as np
import matplotlib.pyplot as plt

# creating wavelength
wavelength = np.linspace(5009,5010,1000)

lambda_0 = 5009.5   #line center of wavelength 
sigma = 0.1  # standard deviation
depth = 0.9  # line depth of the absorption line
# gaussian absorption line profile 
flux = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_0) / sigma) ** 2)

# plot 
plt.plot(wavelength, flux)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('flux')
plt.title('Gaussian Line Profile')
plt.show()
