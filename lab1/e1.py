import numpy as np
import matplotlib.pyplot as plt

c = 3e8
Gt = 1.6
Gr = 1.6
f1 = 900e6
f2 = 2400e6

lambda1 = c / f1
lambda2 = c / f2


def power_drop_db(d, Gt, Gr, lambda_val):
    relative_drop = Gt * Gr * (lambda_val / (4 * np.pi * d)) ** 2
    return 10 * np.log10(relative_drop)


d_a = np.arange(1, 100.25, 0.25)

power_drop_f1_a = power_drop_db(d_a, Gt, Gr, lambda1)
power_drop_f2_a = power_drop_db(d_a, Gt, Gr, lambda2)

d_b = np.linspace(1, 10000, 10000)

power_drop_f1_b = power_drop_db(d_b, Gt, Gr, lambda1)
power_drop_f2_b = power_drop_db(d_b, Gt, Gr, lambda2)

plt.figure(figsize=(12, 7))
plt.plot(d_a, power_drop_f1_a, label="f = 900 MHz", color="blue")
plt.plot(d_a, power_drop_f2_a, label="f = 2400 MHz", color="red")
plt.title("Porównanie spadku mocy dla f1 i f2 (zakres 1-100m)")
plt.xlabel("Odległość [m]")
plt.ylabel("Względny spadek mocy [dB]")
plt.grid(True)
plt.legend()
plt.savefig("wykres_100m_porownanie.png")
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(d_b, power_drop_f1_b, label="f = 900 MHz", color="blue")
plt.plot(d_b, power_drop_f2_b, label="f = 2400 MHz", color="red")
# plt.xscale("log")
plt.title("Porównanie spadku mocy dla f1 i f2 (zakres 1m-10km)")
plt.xlabel("Odległość [m]")
plt.ylabel("Względny spadek mocy [dB]")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig("wykres_10km_porownanie.png")
plt.show()
