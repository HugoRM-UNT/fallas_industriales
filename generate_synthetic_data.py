"""Réplica sintética de AI4I 2020 (SOLO para poder ejecutar el proyecto sin descargar Kaggle).

Reproduce las reglas de generación publicadas por Matzka (2020): mismas columnas, mismos rangos
y los 5 modos de falla (TWF, HDF, PWF, OSF, RNF). NO es el dataset real. Los resultados obtenidos
con este archivo deben regenerarse con el CSV real (ver README).

Uso:  python generate_synthetic_data.py
"""
import hashlib
import numpy as np
import pandas as pd

SEED = 2020
N = 10_000
rng = np.random.default_rng(SEED)


def normalized_walk(n, mean, std):
    w = np.cumsum(rng.normal(size=n))
    w = (w - w.mean()) / w.std()
    return mean + std * w


typ = rng.choice(["L", "M", "H"], size=N, p=[0.6, 0.3, 0.1])
air = np.round(normalized_walk(N, 300, 2.0), 1)
proc = np.round(air + 10 + 0.65 * normalized_walk(N, 0, 1.0), 1)
torque = np.clip(rng.normal(40, 10, N), 3.8, None)
rpm = np.clip(1538 - 15 * (torque - 40) + rng.normal(0, 70, N) + rng.gamma(2, 8, N) - 16, 1168, 2886)
torque = np.round(torque, 1)
rpm = np.round(rpm).astype(int)
wear = np.clip(np.round(rng.normal(106, 62, N)), 0, 253).astype(int)

power = torque * rpm * 2 * np.pi / 60
thr = pd.Series(typ).map({"L": 11000, "M": 12000, "H": 13000}).to_numpy()

twf = ((wear >= 200) & (rng.random(N) < 0.06)).astype(int)
hdf = (((proc - air) < 8.6) & (rpm < 1380)).astype(int)
pwf = ((power < 3500) | (power > 9000)).astype(int)
osf = ((wear * torque) > thr).astype(int)
rnf = (rng.random(N) < 0.001).astype(int)
fail = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

pid = [f"{t}{47000 + i}" for i, t in enumerate(typ)]
df = pd.DataFrame({
    "UDI": np.arange(1, N + 1), "Product ID": pid, "Type": typ,
    "Air temperature [K]": air, "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear,
    "Machine failure": fail, "TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf,
})
df.to_csv("data/ai4i2020.csv", index=False)
h = hashlib.sha256(open("data/ai4i2020.csv", "rb").read()).hexdigest()
open("data/.synthetic_sha256", "w").write(h)
print(df[["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]].sum().to_dict())
print(df.describe().T[["mean", "std", "min", "max"]].round(1))
