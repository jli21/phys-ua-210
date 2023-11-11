import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

# PART A
hdu_list = fits.open("specgrid.fits")
logwave = hdu_list["LOGWAVE"].data
flux = hdu_list["FLUX"].data

wavelength = 10**logwave

plt.figure(figsize=(15, 10))

# Plotting the first five wavelengths 
for i in range(5):
    plt.plot(wavelength, flux[i], label=f'Galaxy {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux ($10^{-17} erg s^{-1} cm^{-2} A^{-1}$)')
plt.legend()
plt.show()

# PART B
normalized_flux = np.zeros_like(flux)  
normalization_factors = np.zeros(flux.shape[0])  

for i in range(flux.shape[0]):
    sum_flux = np.sum(flux[i])
    normalized_flux[i] = flux[i] / sum_flux if sum_flux != 0 else flux[i]  
    normalization_factors[i] = sum_flux # Storing normalization factors

# PART C
mean_spectrum = np.mean(normalized_flux, axis=0)
residuals = normalized_flux - mean_spectrum

# PART D
covariance_matrix = np.dot(residuals.T, residuals) / residuals.shape[0]

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

idx = np.argsort(eigenvalues)[::-1] # Sorting eigenvectors
eigenvectors = eigenvectors[:, idx]

plt.figure(figsize=(15, 10))
for i in range(5):
    plt.plot(wavelength, eigenvectors[:, i], label=f'Eigenvector {i+1}')

plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Amplitude')
plt.title('Eigenvectors from PCA (First 5)')
plt.legend()
plt.show()

# PART E
U, s, Vt = svd(residuals)

eigenvectors_svd = Vt.T

def evaluate_norm(eigen_1, eigen_2):
    for i in range(min(eigen_1.shape[1], eigen_2.shape[1])):
        if np.dot(eigen_1[:, i], eigen_2[:, i]) < 0:
            eigen_2[:, i] = -eigen_2[:, i]
    
    norms = np.linalg.norm(eigen_1 - eigen_2, axis=0)
    return norms

evaluate_norm(eigenvectors_svd, eigenvectors)

# PART F
cond_C = np.linalg.cond(covariance_matrix)
cond_R = np.linalg.cond(residuals)

print("Condition number of C:", cond_C)
print("Condition number of R:", cond_R)

# PART F
n = 5
coefficients = np.dot(residuals, eigenvectors_svd[:, :n])
approx_spectra = np.dot(coefficients, eigenvectors_svd[:, :n].T) + mean_spectrum

# PART H
c_0 = coefficients[:, 0]  
c_1 = coefficients[:, 1]  
c_2 = coefficients[:, 2]  

# Plot c_0 vs c_1
plt.figure(figsize=(10, 8))
plt.scatter(c_0, c_1, alpha=0.5)
plt.xlabel('$c_0$ (First Principal Component)')
plt.ylabel('$c_1$ (Second Principal Component)')
plt.grid(True)
plt.show()

# Plot c_0 vs c_2
plt.figure(figsize=(10, 8))
plt.scatter(c_0, c_2, alpha=0.5)
plt.xlabel('$c_0$ (First Principal Component)')
plt.ylabel('$c_2$ (Third Principal Component)')
plt.grid(True)
plt.show()

# PART I
sq_frac_residuals = []
for Nc in range(1, 21):
    coefficients_Nc = np.dot(residuals, eigenvectors_svd[:, :Nc])
    approx_spectra_Nc = np.dot(coefficients_Nc, eigenvectors_svd[:, :Nc].T) + mean_spectrum

    residuals_Nc = normalized_flux - approx_spectra_Nc
    sq_residuals_Nc = np.square(residuals_Nc)
    sq_norm_flux = np.square(normalized_flux)
    
    sq_frac_residual_Nc = np.sum(sq_residuals_Nc, axis=1) / np.sum(sq_norm_flux, axis=1)
    mean_sq_frac_residual_Nc = np.mean(sq_frac_residual_Nc)
    
    sq_frac_residuals.append(mean_sq_frac_residual_Nc)

plt.plot(range(1, 21), sq_frac_residuals, marker='o')
plt.xlabel('Number of Components (Nc)')
plt.ylabel('Mean Squared Fractional Residuals')
plt.grid(True)
plt.show()

frac_error_Nc_20 = sq_frac_residuals[-1]
print(f"The fractional error for Nc = 20 is {frac_error_Nc_20}")
