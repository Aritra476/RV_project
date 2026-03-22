# Radial Velocity Project

This project is based on understanding how radial velocity is measured using spectral lines.

---

## Theory

Gaussian distribution

λ0 = 5000 (center wavelength)  
σ = 0.05 (standard deviation)  
depth = 0.5 (maximum depth of absorption line)

Flux:

F(λ) = 1 - depth * exp[ -1/2 ((λ - λ0)/σ)^2 ]

Gaussian absorption line.

- linspace create wavelength range from (a, b, N)
- λ0 is the line where absorption is there
- σ is standard deviation between wavelengths
- depth is how strong is absorption

---

## Doppler Shift

λ_shift = λ0 (1 + v/c)

v = radial velocity  
c = speed of light  

Taking:

v = [-2000000, 0, 2000000]

Graph at each radial velocity to see line shift in absorption.

Shift is extremely small.

---

## Noise

There is always noise with flux.

Type of noise: Gaussian noise

F_obs = F_true + N(0, σ)

noise_level = σ

Examples:

0.001 very high precision  
0.01 moderate  
0.05 very noisy  

Noise is added point by point using:

np.random.normal(0, n, len(wavelength))

Flux_noisy = flux + noise

---

## SNR

Noise ~ √N  
SNR ~ N / √N = √N  

Higher photons → higher SNR

---

## RV Precision

σ_v ∝ 1 / (SNR * sharpness * √N_lines)

Also:

σ_v ∝ line width / (line depth * SNR * R)

- higher SNR → better precision  
- sharper line → better precision  
- more lines → better precision  
- larger line width → worse precision  

---

## Fitting

Using Gaussian fit to find λ_center

F(λ) = 1 - A exp[ -1/2 ((λ - λ0)/σ)^2 ]

popt, pcov = curve_fit(gaussian, wavelength, flux_noisy, p0=[depth, lambda_shift, sigma])

popt gives fitted parameters  
pcov gives uncertainty  

---

## Velocity Calculation

v = c (λ_fit - λ0) / λ0

---

## Errors

error in λ = λ_fit - λ_shifted  

error in v = v_fit - v_true  

uncertainty from:

np.sqrt(pcov[1,1])

---

## Observation

- as noise increase → uncertainty increase  
- as line width increase → RV precision decrease  
- shift is very small compared to noise  

