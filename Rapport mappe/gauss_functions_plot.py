import numpy as np
import matplotlib.pyplot as plt

mu_1, var_1 = 1, 0.4
mu_2, var_2 = 2, 1
mu_3, var_3 = -1, 0.8


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Generate x values
x = np.linspace(-4, 5, 1000)

g_1 = gaussian(x, mu_1, var_1)
g_2 = gaussian(x, mu_2, var_2)
g_3 = gaussian(x, mu_3, var_3)

# Plot the Gaussian function
plt.plot(x, g_1, label="Class 1", linewidth=2.5)
plt.plot(x, g_2, label="Class 2", linewidth=2.5)
plt.plot(x, g_3, label="Class 3", linewidth=2.5)

plt.plot([1.75, 1.75], [0, 0.3867], color='r', linestyle='-', linewidth=2.5)
plt.scatter(1.75, 0.3867, color='r', label='$p(x|ω₂)$', s=50)

plt.title('Gaussian Density Functions', fontsize=18)
plt.xlabel('Input sample x', fontsize=15)
plt.ylabel('Probability density', fontsize=15)
plt.grid()
plt.legend(fontsize = 15)
plt.show()


