import numpy as np
import matplotlib.pyplot as plt

c = 3e8

d_short = np.arange(1.0, 100.0 + 0.25, 0.25)
d_long = np.logspace(np.log10(1.0), np.log10(1e4), 2000)


def delay_seconds(d, c=c):
    return d / c


tau_short_s = delay_seconds(d_short)
tau_long_s = delay_seconds(d_long)

tau_short_us = tau_short_s * 1e6
tau_long_us = tau_long_s * 1e6

d_point1 = 100.0
d_point2 = 1e4

tau_p1_us = delay_seconds(d_point1) * 1e6
tau_p2_us = delay_seconds(d_point2) * 1e6

fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

ax = axs[0]
ax.plot(d_short, tau_short_us, label="Opóźnienie (μs)")
ax.set_xlabel("Odległość d [m]")
ax.set_ylabel("Opóźnienie τ [μs]")
ax.set_title("Opóźnienie sygnału — zakres 1–100 m")
ax.grid(True)
ax.legend()

ax.plot(d_point1, tau_p1_us, "ro", label="d = 100 m")
ax.annotate(
    f"100 m\n{tau_p1_us:.3f} μs",
    xy=(d_point1, tau_p1_us),
    xytext=(d_point1 * 0.9, tau_p1_us + 0.03),
    arrowprops=dict(arrowstyle="->"),
    clip_on=False,
)

ax = axs[1]
ax.semilogx(d_long, tau_long_us, label="Opóźnienie (μs)")
ax.set_xlabel("Odległość d [m] (skala log)")
ax.set_ylabel("Opóźnienie τ [μs]")
ax.set_title("Opóźnienie sygnału — zakres 1 m – 10 km")
ax.grid(True, which="both", ls="--")
ax.legend()

ax.plot(d_point1, tau_p1_us, "ro", label="100 m")
ax.plot(d_point2, tau_p2_us, "mo", label="10 km")
ax.annotate(
    f"100 m\n{tau_p1_us:.3f} μs",
    xy=(d_point1, tau_p1_us),
    xytext=(d_point1 * 1.5, tau_p1_us + 1.5),
    arrowprops=dict(arrowstyle="->"),
    clip_on=False,
)
ax.annotate(
    f"10 km\n{tau_p2_us:.3f} μs",
    xy=(d_point2, tau_p2_us),
    xytext=(d_point2 / 3, tau_p2_us + 2.5),
    arrowprops=dict(arrowstyle="->"),
    clip_on=False,
)

plt.tight_layout()

output = "opoznienia_wykresy.png"
fig.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=300)
print(f"Wykres zapisano do pliku: {output}")

print("\nWybrane wartości opóźnień:")
print(f" d = {d_point1:7.1f} m -> τ = {tau_p1_us:9.6f} μs ({tau_p1_us/1000:.6f} ms)")
print(f" d = {d_point2:7.1f} m -> τ = {tau_p2_us:9.6f} μs ({tau_p2_us/1000:.6f} ms)")

for d_example in [1.0, 10.0, 100.0, 1000.0, 1e4]:
    tau_us = delay_seconds(d_example) * 1e6
    print(f" d = {d_example:7.1f} m -> τ = {tau_us:9.6f} μs")
