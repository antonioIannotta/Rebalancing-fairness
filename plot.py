import matplotlib.pyplot as plt
import numpy as np

fairness_algorithms = ['DA', 'FtU', 'EG', 'GS', 'TO']
di_s_min = [0.5409, 0.3809, 0.5102, 0.4103, 0.0000]
di_s_max = [0.5950, 0.4075, 0.6154, 0.4325, 0.0000]

di_r_min = [0.7352, 0.5703, 0.6802, 0.6247, 0.0000]
di_r_max = [0.7486, 0.6308, 0.7442, 0.6376, 0.0000]

dpd_s_min = [0.0155, 0.0197, 0.0174, 0.0149, 0.0000]
dpd_s_max = [0.0171, 0.0242, 0.0265, 0.0163, 0.0000]

dpd_r_min = [0.0051, 0.0063, 0.0064, 0.0046, 0.0000]
dpd_r_max = [0.0053, 0.0083, 0.0092, 0.0052, 0.0000]

min_accuracies = [0.7890, 0.8162, 0.7949, 0.8086, 0.7637]
max_accuracies = [0.7918, 0.8235, 0.8079, 0.8109, 0.7637]

# Calculate the means
mean_accuracies = [(min_val + max_val) / 2 for min_val, max_val in zip(min_accuracies, max_accuracies)]

# Calculate the errors
errors = [(max_val - min_val) / 2 for min_val, max_val in zip(min_accuracies, max_accuracies)]

# Plotting
fig, ax = plt.subplots()
ax.errorbar(mean_accuracies, np.arange(len(fairness_algorithms)), xerr=errors, fmt='o', capsize=5, color='blue')

ax.set_yticks(np.arange(len(fairness_algorithms)))
ax.set_yticklabels(fairness_algorithms)
ax.set_xlabel('Accuracy')
plt.title('Minimum and Maximum Accuracy for Each Fairness Algorithm')
plt.show()