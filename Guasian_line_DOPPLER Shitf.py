# import lib
import numpy as np
import matplotlib.pyplot as plt
c = 299792.458e8 # speed of light in m/s
velocity = [-5000000,-2000000,0,2000000,5000000]# very large values good for visualization of doppler shift, in m/s

# creating wavelength  
wavelength = np.linspace(5009.45,5009.55,1000000)

sigma = 0.005  # standard deviation
depth = 0.9  # line depth of the absorption line


# doppler shifted gaussian absorption line profile 

for v in velocity:
    lambda_shifted = 5009.5* (1 + v/c)   #line center of wavelength 
    flux = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_shifted) / sigma) ** 2)
    plt.plot(wavelength, flux,label = f"v = {v} m/s")
# to plot graph of having differnt radial velocity have to calculate the shifted wavelength

# plot 
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('flux')
plt.title('doppler shift of Gaussian Line Profile')
plt.legend()
plt.xlim(5009.45,5009.55)
plt.show()
