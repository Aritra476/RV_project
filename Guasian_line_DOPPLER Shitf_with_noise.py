# import lib
import numpy as np
import matplotlib.pyplot as plt
c = 299792.458e8 # speed of light in m/s
velocity = [-200000,0,2000000]# very large values good for visualization of doppler shift, in m/s

# creating wavelength  
wavelength = np.linspace(5009.45,5009.55,1000000)

sigma = 0.005  # standard deviation
depth = 0.9  # line depth of the absorption line

noise_level = [0.002] # differnt nosie level to see effect of noise to the original data

# doppler shifted gaussian absorption line profile 

for v in velocity:
    lambda_shifted = 5009.5* (1 + v/c)   #line center of wavelength 
    flux = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_shifted) / sigma) ** 2)
    for n in noise_level:
        noise= np.random.normal(0, n,len(wavelength)) # add noise to the original data
        flux_noisy = flux + noise 
        plt.plot(wavelength, flux,'--',label = "TRUE") # to plot graph of having different radial velocity have to calculate the shifted wavelength
        plt.plot(wavelength, flux_noisy,label =f"noise = {n}") # to plot graph of having different radial velocity have to calculate the shifted wavelength

# plot 
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('flux')
plt.title('doppler shift of Gaussian Line Profile with noise')
plt.legend()
plt.xlim(5009.455,5009.555)
plt.show()
