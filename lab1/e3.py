import numpy as np
import matplotlib.pyplot as plt

c = 3e8

f1 = 900e6
f2 = 2400e6

Gt = 1.6
Gr = 1.6

ht = 1.5
hr = 1.5

Gamma = -0.7

d_short = np.arange(1.0, 100.0 + 0.25, 0.25)
d_long = np.logspace(np.log10(1.0), np.log10(1e4), 2000)


def friis_pr_over_pt(d, f, Gt=Gt, Gr=Gr):
    lam = c / f
    return (Gt * Gr * lam**2) / ((4 * np.pi * d) ** 2)


def two_ray_pr_over_pt(d_array, f, ht=ht, hr=hr, Gamma=Gamma, Gt=Gt, Gr=Gr):
    lam = c / f
    k = 2 * np.pi / lam

    path_direct = np.sqrt(d_array**2 + (ht - hr) ** 2)
    path_reflect = np.sqrt(d_array**2 + (ht + hr) ** 2)

    A_direct = (lam / (4 * np.pi * path_direct)) * np.sqrt(Gt * Gr)
    A_reflec = (lam / (4 * np.pi * path_reflect)) * np.sqrt(Gt * Gr)

    phase_direct = np.exp(-1j * k * path_direct)
    phase_reflec = np.exp(-1j * k * path_reflect)

    E_total = A_direct * phase_direct + Gamma * A_reflec * phase_reflec
    PRPT = np.abs(E_total) ** 2

    return PRPT


def to_db(x):
    return 10.0 * np.log10(x)


prpt_f1_friis_short = friis_pr_over_pt(d_short, f1)
prpt_f1_two_ray_short = two_ray_pr_over_pt(d_short, f1)

prpt_f1_friis_long = friis_pr_over_pt(d_long, f1)
prpt_f1_two_ray_long = two_ray_pr_over_pt(d_long, f1)

prpt_f2_friis_short = friis_pr_over_pt(d_short, f2)
prpt_f2_two_ray_short = two_ray_pr_over_pt(d_short, f2)

prpt_f2_friis_long = friis_pr_over_pt(d_long, f2)
prpt_f2_two_ray_long = two_ray_pr_over_pt(d_long, f2)

prpt_f1_two_ray_long_db = to_db(prpt_f1_two_ray_long)
prpt_f1_friis_long_db = to_db(prpt_f1_friis_long)

prpt_f2_two_ray_long_db = to_db(prpt_f2_two_ray_long)
prpt_f2_friis_long_db = to_db(prpt_f2_friis_long)

plt.rcParams.update({"figure.max_open_warning": 0})
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

ax = axs[0, 0]
ax.plot(d_short, prpt_f1_friis_short, label="Friis (brak odbicia)")
ax.plot(d_short, prpt_f1_two_ray_short, label=f"Dwutorowy (Gamma={Gamma})")
ax.set_xlabel("Odległość d [m]")
ax.set_ylabel("Względny stosunek mocy $P_R/P_T$ (lin.)")
ax.set_title("900 MHz — zakres 1–100 m (model z odbiciem vs Friis)")
ax.grid(True)
ax.legend()

d_point = 100.0
v_friis_100 = friis_pr_over_pt(d_point, f1)
v_tr_100 = two_ray_pr_over_pt(np.array([d_point]), f1)[0]
ax.plot(d_point, v_friis_100, "ro")
ax.plot(d_point, v_tr_100, "mo")

ax = axs[0, 1]
ax.plot(d_short, prpt_f2_friis_short, label="Friis (brak odbicia)")
ax.plot(d_short, prpt_f2_two_ray_short, label=f"Dwutorowy (Gamma={Gamma})")
ax.set_xlabel("Odległość d [m]")
ax.set_ylabel("Względny stosunek mocy $P_R/P_T$ (lin.)")
ax.set_title("2400 MHz — zakres 1–100 m (model z odbiciem vs Friis)")
ax.grid(True)
ax.legend()

v_friis_100_f2 = friis_pr_over_pt(100.0, f2)
v_tr_100_f2 = two_ray_pr_over_pt(np.array([100.0]), f2)[0]
ax.plot(100.0, v_friis_100_f2, "ro")
ax.plot(100.0, v_tr_100_f2, "mo")

ax = axs[1, 0]
ax.semilogx(d_long, prpt_f1_friis_long_db, label="Friis (dB)")
ax.semilogx(d_long, prpt_f1_two_ray_long_db, label=f"Dwutorowy (dB), Gamma={Gamma}")
ax.set_xlabel("Odległość d [m] (skala log)")
ax.set_ylabel("Spadek mocy [dB]")
ax.set_title("900 MHz — zakres 1 m – 10 km (dB)")
ax.grid(True, which="both", ls="--")
ax.legend()

v_friis_100_db = to_db(friis_pr_over_pt(100.0, f1))
v_tr_100_db = to_db(two_ray_pr_over_pt(np.array([100.0]), f1)[0])
v_tr_10k_db = to_db(two_ray_pr_over_pt(np.array([1e4]), f1)[0])
v_friis_10k_db = to_db(friis_pr_over_pt(1e4, f1))

ax.plot(100.0, v_friis_100_db, "ro")
ax.plot(100.0, v_tr_100_db, "mo")
ax.plot(1e4, v_tr_10k_db, "co")
ax.plot(1e4, v_friis_10k_db, "rx")
ax.annotate(
    f"TR 100m\n{v_tr_100_db:.2f} dB",
    xy=(100.0, v_tr_100_db),
    xytext=(150.0, v_tr_100_db + 5),
    arrowprops=dict(arrowstyle="->"),
)
ax.annotate(
    f"TR 10km\n{v_tr_10k_db:.2f} dB",
    xy=(1e4, v_tr_10k_db),
    xytext=(1500.0, v_tr_10k_db - 10),
    arrowprops=dict(arrowstyle="->"),
)

ax = axs[1, 1]
ax.semilogx(d_long, prpt_f2_friis_long_db, label="Friis (dB)")
ax.semilogx(d_long, prpt_f2_two_ray_long_db, label=f"Dwutorowy (dB), Gamma={Gamma}")
ax.set_xlabel("Odległość d [m] (skala log)")
ax.set_ylabel("Spadek mocy [dB]")
ax.set_title("2400 MHz — zakres 1 m – 10 km (dB)")
ax.grid(True, which="both", ls="--")
ax.legend()

v_friis_100_f2_db = to_db(friis_pr_over_pt(100.0, f2))
v_tr_100_f2_db = to_db(two_ray_pr_over_pt(np.array([100.0]), f2)[0])
v_tr_10k_f2_db = to_db(two_ray_pr_over_pt(np.array([1e4]), f2)[0])
v_friis_10k_f2_db = to_db(friis_pr_over_pt(1e4, f2))

ax.plot(100.0, v_friis_100_f2_db, "ro")
ax.plot(100.0, v_tr_100_f2_db, "mo")
ax.plot(1e4, v_tr_10k_f2_db, "co")
ax.plot(1e4, v_friis_10k_f2_db, "rx")
ax.annotate(
    f"TR 100m\n{v_tr_100_f2_db:.2f} dB",
    xy=(100.0, v_tr_100_f2_db),
    xytext=(150.0, v_tr_100_f2_db + 5),
    arrowprops=dict(arrowstyle="->"),
)
ax.annotate(
    f"TR 10km\n{v_tr_10k_f2_db:.2f} dB",
    xy=(1e4, v_tr_10k_f2_db),
    xytext=(1500.0, v_tr_10k_f2_db - 10),
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()
output = "zadanie3_odbicie_wykresy.png"
fig.savefig(output)
print(f"Wykresy zapisano do: {output}")

print(
    "\n== Wybrane wartości PR/PT (lin.) dla modelu z odbiciem i bez (d=100 m, d=10 km) =="
)
for freq, name in [(f1, "900 MHz"), (f2, "2400 MHz")]:
    friis_100 = friis_pr_over_pt(100.0, freq)
    two_100 = two_ray_pr_over_pt(np.array([100.0]), freq)[0]
    friis_10k = friis_pr_over_pt(1e4, freq)
    two_10k = two_ray_pr_over_pt(np.array([1e4]), freq)[0]
    print(f"\nCzęstotliwość: {name}")
    print(
        f" d=100 m -> Friis: {friis_100:.3e} -> {to_db(friis_100):.2f} dB ;  Dwutorowy: {two_100:.3e} -> {to_db(two_100):.2f} dB"
    )
    print(
        f" d=10 km  -> Friis: {friis_10k:.3e} -> {to_db(friis_10k):.2f} dB ;  Dwutorowy: {two_10k:.3e} -> {to_db(two_10k):.2f} dB"
    )
