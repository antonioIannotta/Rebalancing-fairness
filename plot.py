import matplotlib.pyplot as plt
import numpy as np

fairness_algorithms = ['DA', 'FtU', 'EG', 'GS', 'TO']


dpd_r_min = [0.0051, 0.0063, 0.0064, 0.0046, 0.0000]
dpd_r_max = [0.0053, 0.0083, 0.0092, 0.0052, 0.0000]

# Calculate the means
mean_accuracies = [(min_val + max_val) / 2 for min_val, max_val in zip(dpd_r_min, dpd_r_max)]

# Calculate the errors
errors = [(max_val - min_val) / 2 for min_val, max_val in zip(dpd_r_min, dpd_r_max)]

# Plotting
fig, ax = plt.subplots()
ax.errorbar(mean_accuracies, np.arange(len(fairness_algorithms)), xerr=errors, fmt='o', capsize=5, color='blue')

ax.set_yticks(np.arange(len(fairness_algorithms)))
ax.set_yticklabels(fairness_algorithms)
ax.set_xlabel('Demographic Parity Difference Race')
plt.title('Minimum and Maximum Demographic Parity Difference Race for Each Fairness Algorithm')
plt.show()