from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

TX_POWER_DBM: float = 20.0
TX_GAIN_DB: float = 20.0
RX_GAIN_DB: float = 20.0
EXTRA_LOSSES_DB: float = 0.0

D0_M: float = 1.0
GAMMA: float = 4.0

BRICK_DB: float = 8.0
CONCRETE_DB: float = 11.0

INTERNAL_WALL_DB: float = 7.0
EXTERNAL_WALL_DB: float = 9.0
FLOOR_DB: float = 11.0

ITU_N: float = 30.0
ITU_LF: float = 15.0

FREQ_24_GHZ_MHZ: float = 2400.0
FREQ_5_GHZ_MHZ: float = 5000.0


def fspl_db(f_mhz: float, d_m: float) -> float:
    d_m = max(d_m, 1e-3)
    d_km = d_m / 1000.0
    return 32.44 + 20.0 * math.log10(max(f_mhz, 1e-3)) + 20.0 * math.log10(d_km)


def itu_p1238_db(
    f_mhz: float, d_m: float, N: float = ITU_N, n_floors: int = 0, Lf: float = ITU_LF
) -> float:
    d_m = max(d_m, 1.0)
    return (
        20.0 * math.log10(max(f_mhz, 1e-3))
        + N * math.log10(d_m)
        + Lf * float(n_floors)
        - 28.0
    )


def one_slope_db(
    f_mhz: float, d_m: float, gamma: float = GAMMA, d0_m: float = D0_M
) -> float:
    d_m = max(d_m, d0_m)
    L0 = fspl_db(f_mhz, d0_m)
    return L0 + 10.0 * gamma * math.log10(d_m / d0_m)


def motley_keenan_db(
    f_mhz: float,
    d_m: float,
    gamma: float = GAMMA,
    n_brick: int = 0,
    n_concrete: int = 0,
) -> float:
    return (
        one_slope_db(f_mhz, d_m, gamma) + n_brick * BRICK_DB + n_concrete * CONCRETE_DB
    )


def multi_wall_db(
    f_mhz: float,
    d_m: float,
    gamma: float = GAMMA,
    n_internal: int = 0,
    n_external: int = 0,
    n_floors: int = 0,
) -> float:
    return (
        one_slope_db(f_mhz, d_m, gamma)
        + n_internal * INTERNAL_WALL_DB
        + n_external * EXTERNAL_WALL_DB
        + n_floors * FLOOR_DB
    )


SCENARIO_ALIASES = {
    "wewnątrz budynku": "indoors",
    "pomiędzy ścianami": "behind the wall",
    "pomiędzy piętrami": "between floors",
    "na zewnątrz": "outdoors",
}


def normalize_scenario(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().lower())
    return SCENARIO_ALIASES.get(key, name.strip())


def scenario_obstacles(scenariusz: str) -> Dict[str, int]:
    s = normalize_scenario(scenariusz)
    if s == "indoors":
        return dict(n_internal=0, n_external=0, n_floors=0, n_brick=0, n_concrete=0)
    if s == "behind the wall":
        return dict(n_internal=1, n_external=0, n_floors=0, n_brick=1, n_concrete=0)
    if s == "between floors":
        return dict(n_internal=0, n_external=0, n_floors=1, n_brick=0, n_concrete=1)
    if s == "outdoors":
        return dict(n_internal=0, n_external=1, n_floors=0, n_brick=1, n_concrete=0)
    return dict(n_internal=0, n_external=0, n_floors=0, n_brick=0, n_concrete=0)


@dataclass
class Pomiar:
    scenariusz: str
    odleglosc_m: float
    czestotliwosc_MHz: float
    RSSI_dBm: float


def parse_pomiary_txt(path: str) -> List[Pomiar]:
    rows: List[Pomiar] = []
    curr_scen: str | None = None
    curr_dist_m: float | None = None

    header_re = re.compile(r"^\s*(.*?)\s*\(([0-9.]+)\s*m\):?", re.IGNORECASE)
    avg_re = re.compile(r"^\s*śr\.\s*(-?[0-9.]+)\s*dBm", re.IGNORECASE)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            header_m = header_re.search(line)
            if header_m:
                curr_scen = header_m.group(1).strip()
                curr_dist_m = float(header_m.group(2))
                continue

            avg_m = avg_re.search(line)
            if avg_m and curr_scen is not None and curr_dist_m is not None:
                rssi_dbm = float(avg_m.group(1))
                rows.append(Pomiar(curr_scen, curr_dist_m, FREQ_5_GHZ_MHZ, rssi_dbm))
                curr_scen, curr_dist_m = None, None

    return rows


def to_dataframe(rows: List[Pomiar]):
    dicts = [r.__dict__ for r in rows]
    if pd is None:
        return dicts
    return pd.DataFrame(dicts)


def received_power_dbm(L_db: float) -> float:
    return TX_POWER_DBM + TX_GAIN_DB + RX_GAIN_DB - L_db - EXTRA_LOSSES_DB


def compute_models_for_row(row: Pomiar) -> Dict[str, float]:
    f = row.czestotliwosc_MHz
    d = row.odleglosc_m
    obs = scenario_obstacles(row.scenariusz)

    L_fspl = fspl_db(f, d)
    L_itu = itu_p1238_db(f, d, N=ITU_N, n_floors=obs.get("n_floors", 0), Lf=ITU_LF)
    L_os = one_slope_db(f, d, gamma=GAMMA)
    L_mk = motley_keenan_db(
        f,
        d,
        gamma=GAMMA,
        n_brick=obs.get("n_brick", 0),
        n_concrete=obs.get("n_concrete", 0),
    )
    L_mw = multi_wall_db(
        f,
        d,
        gamma=GAMMA,
        n_internal=obs.get("n_internal", 0),
        n_external=obs.get("n_external", 0),
        n_floors=obs.get("n_floors", 0),
    )

    return {
        "FSPL": received_power_dbm(L_fspl),
        "ITU-R P.1238": received_power_dbm(L_itu),
        "One-Slope": received_power_dbm(L_os),
        "Motley-Keenan": received_power_dbm(L_mk),
        "Multi-Wall": received_power_dbm(L_mw),
    }


def evaluate_errors(
    rows: List[Pomiar],
) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    results = []
    per_model_errors_abs: Dict[str, List[float]] = {}
    per_model_errors_sq: Dict[str, List[float]] = {}

    for r in rows:
        preds = compute_models_for_row(r)
        entry = {
            "scenariusz": normalize_scenario(r.scenariusz),
            "odleglosc_m": r.odleglosc_m,
            "czestotliwosc_MHz": r.czestotliwosc_MHz,
            "RSSI_dBm": r.RSSI_dBm,
        }
        entry.update({f"{k}_Po_dBm": v for k, v in preds.items()})
        for m_name, po in preds.items():
            err = po - r.RSSI_dBm
            per_model_errors_abs.setdefault(m_name, []).append(abs(err))
            per_model_errors_sq.setdefault(m_name, []).append(err * err)
        results.append(entry)

    metrics: Dict[str, Dict[str, float]] = {}
    for m_name in per_model_errors_abs:
        mae = sum(per_model_errors_abs[m_name]) / max(
            len(per_model_errors_abs[m_name]), 1
        )
        mse = sum(per_model_errors_sq[m_name]) / max(
            len(per_model_errors_sq[m_name]), 1
        )
        rmse = math.sqrt(mse)
        metrics[m_name] = {"MAE": mae, "RMSE": rmse}

    return results, metrics


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "pomiary.txt")

    if not os.path.exists(data_path):
        print(f"Nie znaleziono pliku z danymi: {data_path}")
        return

    rows = parse_pomiary_txt(data_path)
    if not rows:
        print("Brak poprawnie wczytanych danych z pliku pomiarowego.")
        return

    results, metrics = evaluate_errors(rows)

    print("Metryki modeli (MAE / RMSE):")
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["MAE"]):
        print(f"- {name:15s}  MAE = {m['MAE']:.2f} dB   RMSE = {m['RMSE']:.2f} dB")

    best_model = (
        min(metrics.items(), key=lambda kv: kv[1]["MAE"])[0] if metrics else "brak"
    )
    print(f"\nNajlepiej dopasowany model (wg MAE): {best_model}")

    print("\nSzczegółowe wyniki:")

    model_names = ["FSPL", "ITU-R P.1238", "One-Slope", "Motley-Keenan", "Multi-Wall"]

    header = (
        f"{'Scenariusz':<18s} | {'Odległość':>10s} | {'RSSI (pomiar)':>14s} | "
        + " | ".join(f"{name:>15s}" for name in model_names)
    )
    print(header)
    print("-" * len(header))

    for res in results:
        row_str = f"{res['scenariusz']:<18s} | {res['odleglosc_m']:>8.1f} m | {res['RSSI_dBm']:>11.1f} dBm | "
        model_values = []
        for name in model_names:
            val = res.get(f"{name}_Po_dBm", float("nan"))
            model_values.append(f"{val:>12.1f} dBm")

        row_str += " | ".join(model_values)
        print(row_str)


if __name__ == "__main__":
    main()
