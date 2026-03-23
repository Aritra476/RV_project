# ================= IMPORTS =================
import numpy as np
import matplotlib.pyplot as plt

# ================= CONSTANTS =================
c = 299792458 # speed of light

# ================= LINE PARAMETERS =================
line_center = np.array([5009.45, 5009.40, 5009.50, 5009.55, 5009.60])
line_depth = np.array([0.8, 0.6, 0.4, 0.9, 0.7])
sigmas = np.array([0.01, 0.008, 0.012, 0.015, 0.011])

v_true = 2e4
v_grid = np.linspace(-3e4, 3e4, 1500)

# ================= FUNCTION =================
def measure_rv_single_order(log_lambda, observed_clean,
                           center, depths, sigmas,
                           noise_level, n_trials=10):

    rv_results = []

    # precompute
    x0_list = np.log(center)
    sigma_log_list = sigmas / center

    for _ in range(n_trials):

        # ================= PHOTON NOISE =================
        exposure = 1 / (noise_level**2 + 1e-12)
        observed_noisy = np.random.poisson(observed_clean * exposure) / exposure

        # ================= CONTINUUM (realistic effect) =================
        lam = np.exp(log_lambda)
        continuum = 1 + 0.02 * (lam - np.mean(lam))
        observed_noisy *= continuum

        # continuum normalization
        smooth = np.convolve(observed_noisy, np.ones(50)/50, mode='same')
        observed_noisy /= (smooth + 1e-12)

        # ================= MASK (bad regions + random pixels) =================
        mask = np.ones_like(log_lambda)

        # remove one spectral region
        mask[(lam > 5009.48) & (lam < 5009.52)] = 0

        # random bad pixels
        bad_pixels = np.random.choice(len(mask), size=20, replace=False)
        mask[bad_pixels] = 0

        # ================= OPTIMAL WEIGHTS (CR bound physics) =================
        dF = np.gradient(observed_noisy, log_lambda)

        weights = mask * (dF**2) / (observed_noisy + 1e-6)

        # CRITICAL FIX → avoid zero-weight collapse
        if np.sum(weights) == 0:
            continue

        # ================= NORMALIZATION =================
        w_mean = np.sum(observed_noisy * weights) / np.sum(weights)
        observed_norm = observed_noisy - w_mean

        ccf_values = []

        for v in v_grid:

            template = np.ones_like(log_lambda)

            for x0, d, sigma_log in zip(x0_list, depths, sigma_log_list):

                # ================= TEMPLATE MISMATCH =================
                d_temp = d * 0.9
                sigma_temp = sigma_log * 1.2

                x_shifted = x0 + v / c

                line = 1 - d_temp * np.exp(
                    -0.5 * ((log_lambda - x_shifted) / sigma_temp)**2
                )

                template *= line

            # normalize template
            w_mean_temp = np.sum(template * weights) / np.sum(weights)
            template_norm = template - w_mean_temp

            # ================= WEIGHTED CCF =================
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
            v_measured = v2 + (c1 - c3)/(2*(c1 - 2*c2 + c3)) * dv
        else:
            v_measured = v_grid[i]

        rv_results.append(v_measured)

    return np.mean(rv_results), np.std(rv_results)


# ================= MULTI-ORDER =================

orders = [
    (5008.5, 5010.5),
    (5010.5, 5012.5),
    (5012.5, 5014.5)
]

order_data = []

for wmin, wmax in orders:

    wavelength = np.linspace(wmin, wmax, 2000)
    log_lambda = np.log(wavelength)

    observed_clean = np.ones_like(log_lambda)

    # include only lines inside order
    mask = (line_center >= wmin) & (line_center <= wmax)

    if np.sum(mask) == 0:
        continue

    for ic, d, s in zip(line_center[mask],
                        line_depth[mask],
                        sigmas[mask]):

        x0 = np.log(ic)
        sigma_log = s / ic
        x_shifted = x0 + v_true / c

        line = 1 - d * np.exp(
            -0.5 * ((log_lambda - x_shifted) / sigma_log)**2
        )

        observed_clean *= line

    order_data.append((log_lambda, observed_clean, mask))


# ================= RV PER ORDER =================

v_list, sigma_list = [], []

for log_lambda_k, obs_k, mask in order_data:

    v_k, sigma_k = measure_rv_single_order(
        log_lambda_k,
        obs_k,
        line_center[mask],
        line_depth[mask],
        sigmas[mask],
        noise_level=0.01,
        n_trials=10
    )

    # reject extremely bad orders (stability fix)
    if sigma_k < 100:
        v_list.append(v_k)
        sigma_list.append(sigma_k)

v_list = np.array(v_list)
sigma_list = np.array(sigma_list)

print("Order RVs:", v_list)
print("Order sigmas:", sigma_list)


# ================= COMBINE ORDERS =================

weights = 1 / (sigma_list**2 + 1e-6)

v_final = np.sum(v_list * weights) / np.sum(weights)
sigma_final = np.sqrt(1 / np.sum(weights))

print("\nFinal RV:", v_final)
print("Final Precision:", sigma_final)