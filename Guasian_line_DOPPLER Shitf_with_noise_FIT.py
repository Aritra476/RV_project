# import lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

c = 299792.458e8 # speed of light in m/s
velocity = [-2000000,0,2000000]# very large values good for visualization of doppler shift, in m/s
lambda0 = 5009.5 # line center of wavelength

# creating wavelength  
wavelength = np.linspace(5009.445,5009.555,1000000)

sigma = 0.005  # standard deviation
depth = 0.9  # line depth of the absorption line

noise_level = [0.002] # differnt nosie level to see effect of noise to the original data

# doppler shifted gaussian absorption line profile 

for v in velocity:
    lambda_shifted = 5009.500* (1 + v/c)   #line center of wavelength 
    flux = 1 - depth * np.exp(-0.5 * ((wavelength - lambda_shifted) / sigma) ** 2)
    for n in noise_level:
        noise= np.random.normal(0, n,len(wavelength)) # add noise to the original data
        flux_noisy = flux + noise 
        plt.plot(wavelength, flux,'--',label = "TRUE") # to plot graph of having different radial velocity have to calculate the shifted wavelength
        plt.plot(wavelength, flux_noisy,label =f"noise = {n}") # to plot graph of having different radial velocity have to calculate the shifted wavelength

def gaussian(w, depth, lambda0, sigma):
    return 1 - depth * np.exp(-0.5 * ((w - lambda0) / sigma) ** 2)
popt , pcov = curve_fit(gaussian, wavelength, flux_noisy, p0=[depth, lambda_shifted, sigma])
A_fit, lambda0_fit, sigma_fit = popt

#popt means the optimal parameters 
# pcov means the covariance of the matrix of the parameters gives uncertainty in the fitted parameters (how relaible the fit is )

v_fit = (lambda0_fit - lambda0) / lambda0 * c

errors = np.sqrt(np.diag(pcov))

delta_lambda = lambda0_fit - lambda_shifted
delta_v = delta_lambda / lambda0 * c

print('True lammbda :', lambda_shifted)
print('Fitted lambda :', lambda0_fit)
print('error in lambda :',(lambda0_fit-lambda_shifted))


v_true = (lambda_shifted - lambda0) / lambda0 * c
v_fit = (lambda0_fit - lambda0) / lambda0 * c
print ('True velocity :', v_true)
print ('Fitted velocity :', v_fit)
print('error in velocity :', (v_fit - v_true))


# plot 
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('flux')
plt.title('doppler shift of Gaussian Line Profile with noise and fit')
plt.legend()
plt.xlim(5009.455,5009.555)
plt.show()