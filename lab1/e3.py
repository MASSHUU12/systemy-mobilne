import numpy as np
import matplotlib.pyplot as plt

c = 3e8
Gt = 1.6
Gr = 1.6
f1 = 900e6
f2 = 2400e6
h1 = 30
h2 = 3

lambda1 = c / f1
lambda2 = c / f2

# Case a: 1m - 100m range
d_short_range = np.arange(1, 100.5, 0.5)
d1_a = np.sqrt((h1 - h2) ** 2 + d_short_range**2)
d2_a = np.sqrt((h1 + h2) ** 2 + d_short_range**2)

phi1_f1_a = -2 * np.pi * f1 * d1_a / c
phi2_f1_a = -2 * np.pi * f1 * d2_a / c
phi1_f2_a = -2 * np.pi * f2 * d1_a / c
phi2_f2_a = -2 * np.pi * f2 * d2_a / c

prpt_f1_a = (
    Gt
    * Gr
    * (lambda1 / (4 * np.pi)) ** 2
    * np.abs(np.exp(1j * phi1_f1_a) / d1_a - np.exp(1j * phi2_f1_a) / d2_a) ** 2
)
prpt_f2_a = (
    Gt
    * Gr
    * (lambda2 / (4 * np.pi)) ** 2
    * np.abs(np.exp(1j * phi1_f2_a) / d1_a - np.exp(1j * phi2_f2_a) / d2_a) ** 2
)

# Case b: 1m - 10km range
d_long_range = np.logspace(0, 4, 600)
d1_b = np.sqrt((h1 - h2) ** 2 + d_long_range**2)
d2_b = np.sqrt((h1 + h2) ** 2 + d_long_range**2)

phi1_f1_b = -2 * np.pi * f1 * d1_b / c
phi2_f1_b = -2 * np.pi * f1 * d2_b / c
phi1_f2_b = -2 * np.pi * f2 * d1_b / c
phi2_f2_b = -2 * np.pi * f2 * d2_b / c

prpt_f1_b = (
    Gt
    * Gr
    * (lambda1 / (4 * np.pi)) ** 2
    * np.abs(np.exp(1j * phi1_f1_b) / d1_b - np.exp(1j * phi2_f1_b) / d2_b) ** 2
)
prpt_f2_b = (
    Gt
    * Gr
    * (lambda2 / (4 * np.pi)) ** 2
    * np.abs(np.exp(1j * phi1_f2_b) / d1_b - np.exp(1j * phi2_f2_b) / d2_b) ** 2
)

prpt_f1_a_db = 10 * np.log10(prpt_f1_a)
prpt_f2_a_db = 10 * np.log10(prpt_f2_a)
prpt_f1_b_db = 10 * np.log10(prpt_f1_b)
prpt_f2_b_db = 10 * np.log10(prpt_f2_b)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Two-Ray Model", fontsize=16)

ax1.plot(d_short_range, prpt_f1_a_db, "b", label="f_1 = 900 MHz", linewidth=1.2)
ax1.plot(d_short_range, prpt_f2_a_db, "r", label="f_2 = 2.4 GHz", linewidth=1.2)
ax1.grid(True)
ax1.set_xlabel("Distance d [m]")
ax1.set_ylabel("10log$_{10}$(P$_R$/P$_T$) [dB]")
ax1.set_title("Two-Ray Model: 1–100 m")
ax1.legend(loc="best")

ax2.semilogx(d_long_range, prpt_f1_b_db, "b", label="f_1 = 900 MHz", linewidth=1.2)
ax2.semilogx(d_long_range, prpt_f2_b_db, "r", label="f_2 = 2.4 GHz", linewidth=1.2)
ax2.grid(True)
ax2.set_xlabel("Distance d [m] (log scale)")
ax2.set_ylabel("10log$_{10}$(P$_R$/P$_T$) [dB]")
ax2.set_title("Two-Ray Model: 1 m – 10 km")
ax2.legend(loc="best")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("task3.png")
plt.show()
