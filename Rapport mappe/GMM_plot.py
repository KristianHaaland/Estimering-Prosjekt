import numpy as np
import matplotlib.pyplot as plt

mu_1, var_1, w_1 = 3, 1.5, 0.33
mu_2, var_2, w_2= 5.5, 1.2, 0.33
mu_3, var_3, w_3 = -1, 2, 0.33

def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Generate x values
x = np.linspace(-7, 10, 1000)

g_1 = w_1*gaussian(x, mu_1, var_1)
g_2 = w_2*gaussian(x, mu_2, var_2)
g_3 = w_3*gaussian(x, mu_3, var_3)
gmm = g_1 + g_2 + g_3


# Plot the Gaussian function
plt.plot(x, g_1, label="$c_{i1*}N(\\mu_{i1}, \\Sigma_{i1})$")
plt.fill_between(x, g_1, color='skyblue', alpha=0.3)
plt.plot(x, g_2, label="$c_{i2*}N(\\mu_{i2}, \\Sigma_{i2})$")
plt.fill_between(x, g_2, color='orange', alpha=0.3)
plt.plot(x, g_3, label="$c_{i2*}N(\\mu_{i2}, \\Sigma_{i2})$")
plt.fill_between(x, g_3, color='green', alpha=0.3)
plt.plot(x, gmm, label="p(x|ωᵢ)", linewidth=2.5)

plt.title('Gaussian Mixture Model', fontsize=18)
plt.xlabel('Input value x', fontsize=15)
plt.ylabel('Probability density', fontsize=15)
plt.grid()
plt.legend(fontsize = 15)
plt.show()
