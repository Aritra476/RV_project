# imports
import numpy as np
import matplotlib.pyplot as plt

# constants
c = 299792458  # speed of light in m/s

# wavelength grid space
wavelength = np.linspace(5008.5, 5010.5, 2000)
log_lambda = np.log(wavelength)

# line parameters
line_center = [5009.45, 5009.40, 5009.50, 5009.55, 5009.60]
line_depths = [0.6, 0.5, 0.7, 0.9, 0.8]
sigmas = [0.01, 0.008, 0.012, 0.015, 0.011]

v_true = 2e4  # m/s

# velocity grid
v_grid = np.linspace(-3e4, 3e4, 1500)


# ================= FUNCTION =================
def measure_rv_std(noise_level, center, depths, sigmas, n_trials=50):

    rv_results = []

    center = np.array(center)
    depths = np.array(depths)
    sigmas = np.array(sigmas)

    x0_list = np.log(center)
    sigma_log_list = sigmas / center

    for _ in range(n_trials):

        # ================= BUILD OBSERVED SPECTRUM (FIXED) =================
        # must use SAME lines as template (important for physics consistency)
        observed_clean_local = np.ones_like(log_lambda)

        for x0_i, d, sigma_log_i in zip(x0_list, depths, sigma_log_list):
            x_shifted_i = x0_i + v_true / c
            line_i = 1 - d * np.exp(
                -0.5 * ((log_lambda - x_shifted_i) / sigma_log_i) ** 2
            )
            observed_clean_local *= line_i

        # ================= NOISE =================
        # poisson noise scaled with exposure ~ 1/sigma^2
        exposure = 1 / (noise_level**2 + 1e-12)
        observed_noisy = np.random.poisson(observed_clean_local * exposure) / exposure

        # ================= WEIGHTS =================
        # photon noise → variance ~ flux → weight ~ 1/flux
        weights = 1 / (observed_clean_local + 1e-6)

        # weighted mean subtraction
        w_mean_obs = np.sum(observed_noisy * weights) / np.sum(weights)
        observed_norm = observed_noisy - w_mean_obs

        # ================= CCF =================
        ccf_values = []

        for v in v_grid:

            shifted_template = np.ones_like(log_lambda)

            for x0_i, d, sigma_log_i in zip(x0_list, depths, sigma_log_list):
                x_shifted_i = x0_i + v / c
                line_i = 1 - d * np.exp(
                    -0.5 * ((log_lambda - x_shifted_i) / sigma_log_i) ** 2
                )
                shifted_template *= line_i

            # weighted template normalization
            w_mean_temp = np.sum(shifted_template * weights) / np.sum(weights)
            template_norm = shifted_template - w_mean_temp

            # weighted cross-correlation (MLE estimator)
            num = np.sum(weights * observed_norm * template_norm)
            denom = np.sqrt(
                np.sum(weights * observed_norm**2) *
                np.sum(weights * template_norm**2)
            )

            ccf_values.append(num / (denom + 1e-12))

        # ================= PEAK DETECTION =================
        i = np.argmax(ccf_values)

        if 0 < i < len(ccf_values) - 1:
            v1, v2, v3 = v_grid[i-1], v_grid[i], v_grid[i+1]
            c1, c2, c3 = ccf_values[i-1], ccf_values[i], ccf_values[i+1]

            dv = v2 - v1
            v_measured = v2 + (c1 - c3) / (2 * (c1 - 2*c2 + c3)) * dv
        else:
            v_measured = v_grid[i]

        rv_results.append(v_measured)

    return np.std(rv_results)


# ================= NOISE SCALING =================
noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
rv_std_noise = []

for nl in noise_levels:
    std = measure_rv_std(nl, line_center, line_depths, sigmas, n_trials=10)
    rv_std_noise.append(std)
    print(f"Noise Level: {nl:.3f}, RV Std: {std:.2f} m/s")

plt.plot(noise_levels, rv_std_noise, marker='o')
plt.xlabel('Noise Level')
plt.ylabel('RV Std (m/s)')
plt.title('RV Precision vs Noise')
plt.show()


# ================= LINE COUNT SCALING =================
line_counts = [1, 2, 3, 4, 5]
rv_std_line = []

for n in line_counts:
    ic = line_center[:n]
    idp = line_depths[:n]
    sg = sigmas[:n]

    std = measure_rv_std(0.01, ic, idp, sg, n_trials=10)
    rv_std_line.append(std)

    print(f"Line Count: {n}, RV Std: {std:.2f} m/s")

plt.plot(line_counts, rv_std_line, marker='o')
plt.xlabel('Number of Lines')
plt.ylabel('RV Std (m/s)')
plt.title('RV Precision vs Number of Lines')
plt.show()


# ================= Q FACTOR (FULL SPECTRUM) =================
# use full 5-line spectrum for reference

observed_clean = np.ones_like(log_lambda)
for ic, d, s in zip(line_center, line_depths, sigmas):
    x0_i = np.log(ic)
    sigma_log_i = s / ic
    x_shifted_i = x0_i + v_true / c

    line_i = 1 - d * np.exp(
        -0.5 * ((log_lambda - x_shifted_i) / sigma_log_i) ** 2
    )
    observed_clean *= line_i

dl_dx = np.gradient(observed_clean, log_lambda)

Q_sq = np.sum(dl_dx**2 / observed_clean)
Q = np.sqrt(Q_sq)

print(f"Q factor: {Q:.3f}")

# predicted RV precision
N_phot = np.sum(observed_clean)
sigma_rv_pred = c / (Q * np.sqrt(N_phot))

print(f"Predicted RV Precision: {sigma_rv_pred:.2f} m/s")