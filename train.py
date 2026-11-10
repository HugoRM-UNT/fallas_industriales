"""Trabajo aplicativo Grupo 09 - Detección de fallas en equipos industriales.

Compara Decision Tree, Random Forest y Gradient Boosting (clasificación binaria) sobre el MISMO
problema, partición y folds. Ejecuta las 16 etapas de la guía, guarda figuras, el Pipeline ganador
(joblib) y todos los resultados en model/results.json (que luego consumen el informe, las
diapositivas y la app Streamlit).

Uso:
    python train.py                 # configuración completa (20 configs aleatorias por modelo)
    python train.py --n-iter 5      # prueba rápida
    python train.py --data ruta.csv # otro CSV
"""
import argparse
import hashlib
import json
import os
import platform
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, make_scorer, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ configuración global
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/ai4i2020.csv")
ap.add_argument("--n-iter", type=int, default=20)
args = ap.parse_args()

SEED = 42                 # semilla única del proyecto
N_ITER = args.n_iter      # presupuesto de búsqueda: configuraciones aleatorias por modelo
N_FOLDS = 5
TEST_SIZE = 0.20
C_FN, C_FA = 10.0, 1.0    # costo relativo: falla no detectada vs. falsa alarma (supuesto del grupo)
PREC_FLOOR = 0.25         # piso de precisión usado SOLO en el análisis de sensibilidad (criterio recall)
FEATS_NUM = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
FEATS_CAT = ["type"]
RENAME = {"Air temperature [K]": "air_temp", "Process temperature [K]": "process_temp",
          "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
          "Type": "type", "Machine failure": "failure"}
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
COL = {"normal": "#4C78A8", "falla": "#E45756", "Decision Tree": "#F58518",
       "Random Forest": "#54A24B", "Gradient Boosting": "#B279A2"}

os.makedirs("model", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def clean(o):
    """Convierte tipos numpy a tipos JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


R = {}  # resultados -> results.json
t_start = time.time()

# =================================================================== ETAPA 02: carga
raw = pd.read_csv(args.data, encoding="utf-8-sig")
sha = hashlib.sha256(open(args.data, "rb").read()).hexdigest()
syn_file = os.path.join(os.path.dirname(args.data), ".synthetic_sha256")
is_synth = os.path.exists(syn_file) and open(syn_file).read().strip() == sha
info = json.load(open("dataset_info.json")) if os.path.exists("dataset_info.json") else {}
R["dataset"] = {"info": info, "is_synthetic": is_synth, "sha256": sha, "file": args.data,
                "n_rows": len(raw), "n_cols": raw.shape[1], "columns": list(raw.columns),
                "dtypes": {c: str(t) for c, t in raw.dtypes.items()}}
print(f"[02] Cargado {args.data}: {raw.shape} | sintético={is_synth}")

# =================================================================== ETAPA 03: exploración
missing = raw.isna().sum()
R["eda"] = {
    "missing_total": int(missing.sum()),
    "missing_by_col": missing.to_dict(),
    "duplicated_rows": int(raw.duplicated().sum()),
    "duplicated_ids": int(raw["Product ID"].duplicated().sum()) if "Product ID" in raw else None,
    "class_counts": raw["Machine failure"].value_counts().sort_index().to_dict(),
    "fail_rate": float(raw["Machine failure"].mean()),
    "fail_rate_by_type": raw.groupby("Type")["Machine failure"].mean().to_dict(),
    "type_counts": raw["Type"].value_counts().to_dict(),
    "describe": raw[list(RENAME)[:5]].describe().round(2).to_dict(),
    "mean_by_class": raw.groupby("Machine failure")[list(RENAME)[:5]].mean().round(2).to_dict(),
}
num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]"]
short = ["Temp. aire", "Temp. proceso", "Velocidad", "Torque", "Desgaste"]

# gráfico 1: balance de clases + tasa por tipo
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
cc = raw["Machine failure"].value_counts().sort_index()
ax[0].bar(["Normal (0)", "Falla (1)"], cc.values, color=[COL["normal"], COL["falla"]])
for i, v in enumerate(cc.values):
    ax[0].text(i, v, f"{v}\n({v / len(raw):.1%})", ha="center", va="bottom")
ax[0].set_title("Balance de clases"); ax[0].set_ylim(0, cc.max() * 1.2)
ft = raw.groupby("Type")["Machine failure"].mean().reindex(["L", "M", "H"]) * 100
ax[1].bar(ft.index, ft.values, color="#8c8c8c")
for i, v in enumerate(ft.values):
    ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
ax[1].set_title("Tasa de falla por tipo de producto"); ax[1].set_ylabel("% fallas")
savefig("fig01_clases")

# gráfico 2: distribuciones por clase
fig, axs = plt.subplots(2, 3, figsize=(9, 5))
for a, c, s in zip(axs.ravel(), num_cols, short):
    for k, nm in [(0, "normal"), (1, "falla")]:
        a.hist(raw.loc[raw["Machine failure"] == k, c], bins=30, alpha=.6, density=True,
               color=COL[nm], label=nm)
    a.set_title(s); a.set_yticks([])
axs[1, 2].axis("off"); axs[0, 0].legend()
savefig("fig02_distribuciones")

# gráfico 3: correlaciones
corr = raw[num_cols + ["Machine failure"]].corr()
corr.columns = corr.index = short + ["Falla"]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index); ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, fraction=.046); ax.set_title("Correlación (Pearson)")
savefig("fig03_correlacion")
R["eda"]["corr_with_target"] = corr["Falla"].drop("Falla").round(3).to_dict()
R["eda"]["corr_rpm_torque"] = float(corr.loc["Velocidad", "Torque"])

# =================================================================== ETAPA 04: limpieza / fuga
any_mode = raw[MODES].max(axis=1)
leak_ok = bool((any_mode == raw["Machine failure"]).all())
R["cleaning"] = {
    "target_equals_any_mode": leak_ok,
    "n_fail_without_mode": int(((any_mode == 0) & (raw["Machine failure"] == 1)).sum()),
    "n_mode_without_fail": int(((any_mode == 1) & (raw["Machine failure"] == 0)).sum()),
    "mode_counts": raw[MODES].sum().to_dict(),
    "dropped": ["UDI", "Product ID"] + MODES,
    "dup_feature_rows": int(raw[num_cols + ["Type"]].duplicated().sum()),
    "unique_product_ids": bool(raw["Product ID"].is_unique),
    "type_matches_product_id": bool((raw["Product ID"].str[0] == raw["Type"]).all()),
}
print(f"[04] target == OR(modos): {leak_ok}  -> columnas de causa excluidas")

df = raw.rename(columns=RENAME).drop(columns=["UDI", "Product ID"] + MODES)
for c in FEATS_NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =================================================================== ETAPA 05: X e y
X, y = df[FEATS_NUM + FEATS_CAT], df["failure"].astype(int)
R["xy"] = {"features_num": FEATS_NUM, "features_cat": FEATS_CAT, "target": "failure"}

# =================================================================== ETAPA 06: Train/Test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
R["split"] = {"seed": SEED, "test_size": TEST_SIZE, "n_train": len(X_tr), "n_test": len(X_te),
              "pos_train": int(y_tr.sum()), "pos_test": int(y_te.sum()),
              "rate_train": float(y_tr.mean()), "rate_test": float(y_te.mean())}
cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)   # MISMOS folds para los 3 modelos
sw_tr = compute_sample_weight("balanced", y_tr)                  # mismo tratamiento del desbalance
FIT = {"model__sample_weight": sw_tr}


# =================================================================== ETAPAS 07-08: Pipelines
def make_pipe(est):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), FEATS_NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               categories=[["L", "M", "H"]]))]), FEATS_CAT),
    ])
    return Pipeline([("pre", pre), ("model", est)])


MODELS = {
    "Decision Tree": make_pipe(DecisionTreeClassifier(random_state=SEED)),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting": make_pipe(GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
}
SPACES = {
    "Decision Tree": {"model__max_depth": [3, 4, 5, 6, 8, 10, 15, None],
                      "model__min_samples_split": randint(2, 40),
                      "model__min_samples_leaf": randint(1, 25),
                      "model__max_features": [None, "sqrt", 0.7],
                      "model__ccp_alpha": uniform(0, 0.01)},
    "Random Forest": {"model__n_estimators": randint(100, 251),
                      "model__max_depth": [6, 8, 10, 15, None],
                      "model__min_samples_split": randint(2, 30),
                      "model__min_samples_leaf": randint(1, 15),
                      "model__max_features": ["sqrt", "log2", 0.7],
                      "model__bootstrap": [True, False]},
    "Gradient Boosting": {"model__n_estimators": randint(60, 261),
                          "model__learning_rate": uniform(0.02, 0.28),
                          "model__max_depth": randint(2, 5),
                          "model__min_samples_split": randint(2, 30),
                          "model__min_samples_leaf": randint(1, 15),
                          "model__subsample": uniform(0.6, 0.4)},
}
SCORING = {"recall": "recall", "precision": "precision", "pr_auc": "average_precision",
           "roc_auc": "roc_auc", "f2": make_scorer(fbeta_score, beta=2),
           "accuracy": "accuracy"}
R["config"] = {"seed": SEED, "n_iter": N_ITER, "n_folds": N_FOLDS, "c_fn": C_FN, "c_fa": C_FA,
               "precision_floor": PREC_FLOOR, "primary_metric": "PR-AUC (average precision) de la clase falla",
               "secondary_metrics": ["recall", "precision", "f2", "roc_auc", "accuracy"],
               "imbalance": "sample_weight='balanced' idéntico en los tres modelos",
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "pandas": pd.__version__, "numpy": np.__version__,
                            "scipy": scipy.__version__, "joblib": joblib.__version__},
               "spaces": {m: {k.replace("model__", ""): (str(v) if not hasattr(v, "dist") else
                                                          f"{v.dist.name}{v.args}")
                              for k, v in s.items()} for m, s in SPACES.items()}}


def complexity(pipe):
    m = pipe.named_steps["model"]
    if isinstance(m, DecisionTreeClassifier):
        return {"trees": 1, "total_nodes": int(m.tree_.node_count), "max_depth": int(m.get_depth())}
    trees = [e for e in np.ravel(m.estimators_)]
    return {"trees": len(trees), "total_nodes": int(sum(t.tree_.node_count for t in trees)),
            "max_depth": int(max(t.get_depth() for t in trees))}


def pred_ms_per_1000(pipe):
    t0 = time.perf_counter()
    for _ in range(3):
        pipe.predict_proba(X_te.iloc[:1000])
    return (time.perf_counter() - t0) / 3 * 1000 * (1000 / min(1000, len(X_te)))


# =================================================================== ETAPAS 09-10: CV base
print("[09-10] CV base ...")
CACHE = "model/cache"          # checkpoints: si se interrumpe, al reejecutar se retoma
os.makedirs(CACHE, exist_ok=True)
KEY = f"{sha[:8]}_{N_ITER}_{SEED}"


def cached(label, fn):
    p = f"{CACHE}/{label}_{KEY}.pkl"
    if os.path.exists(p):
        return joblib.load(p)
    out = fn()
    joblib.dump(out, p)
    return out


def run_base():
    base, base_fold = {}, {}
    for name, pipe in MODELS.items():
        r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING, params=FIT, return_train_score=True)
        fitted = clone(pipe).fit(X_tr, y_tr, **FIT)
        base[name] = {k: float(np.mean(r[f"test_{k}"])) for k in SCORING}
        base[name].update({f"{k}_std": float(np.std(r[f"test_{k}"])) for k in SCORING})
        base[name].update({"train_pr_auc": float(np.mean(r["train_pr_auc"])),
                           "train_recall": float(np.mean(r["train_recall"])),
                           "fit_time_s": float(np.mean(r["fit_time"])), "complexity": complexity(fitted)})
        base_fold[name] = {"recall": r["test_recall"].tolist(), "pr_auc": r["test_pr_auc"].tolist()}
        print(f"   {name}: recall={base[name]['recall']:.3f}+-{base[name]['recall_std']:.3f} "
              f"pr_auc={base[name]['pr_auc']:.3f}")
    return base, base_fold


base, base_fold = cached("base", run_base)
R["base"] = base


# =================================================================== ETAPA 11: optimización
def alt_recall_with_floor(res):
    """Criterio ALTERNATIVO (sensibilidad): mejor recall CV entre configs con precisión >= PREC_FLOOR."""
    rec, prec, pr = (np.array(res[f"mean_test_{k}"]) for k in ("recall", "precision", "pr_auc"))
    ok = np.where(prec >= PREC_FLOOR)[0]
    if len(ok) == 0:
        return int(np.argmax(pr))
    return int(max(ok, key=lambda i: (round(rec[i], 4), pr[i])))


print("[11] Optimización (RandomizedSearchCV) ...")
tuned, best, fold_scores, search_pts, alt = {}, {}, {}, {}, {}


def run_search(name):
    pipe = MODELS[name]
    t0 = time.time()
    rs = RandomizedSearchCV(pipe, SPACES[name], n_iter=N_ITER, cv=cv, scoring=SCORING,
                            refit="pr_auc", random_state=SEED, n_jobs=-1)
    rs.fit(X_tr, y_tr, **FIT)               # ETAPA 12: refit automático en TODO Train
    elapsed = time.time() - t0
    i = rs.best_index_
    res = rs.cv_results_
    bp = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
          for k, v in rs.best_params_.items()}
    T_ = {
        "best_params": bp, "best_index": int(i),
        "recall_cv": float(res["mean_test_recall"][i]), "recall_cv_std": float(res["std_test_recall"][i]),
        "precision_cv": float(res["mean_test_precision"][i]),
        "pr_auc_cv": float(res["mean_test_pr_auc"][i]), "pr_auc_cv_std": float(res["std_test_pr_auc"][i]),
        "f2_cv": float(res["mean_test_f2"][i]), "roc_auc_cv": float(res["mean_test_roc_auc"][i]),
        "accuracy_cv": float(res["mean_test_accuracy"][i]),
        "n_configs": int(len(res["params"])),
        "n_configs_ok": int((np.array(res["mean_test_precision"]) >= PREC_FLOOR).sum()),
        "search_time_s": elapsed, "mean_fit_time_s": float(res["mean_fit_time"][i]),
        "complexity": complexity(rs.best_estimator_),
        "pred_ms_per_1000": pred_ms_per_1000(rs.best_estimator_),
    }
    j = alt_recall_with_floor(res)
    A_ = {"index": j, "recall_cv": float(res["mean_test_recall"][j]),
                 "precision_cv": float(res["mean_test_precision"][j]),
                 "pr_auc_cv": float(res["mean_test_pr_auc"][j]), "same_as_chosen": bool(j == i)}
    F_ = {m: [float(res[f"split{k}_test_{m}"][i]) for k in range(N_FOLDS)]
                         for m in ("recall", "pr_auc", "precision")}
    S_ = {"recall": [float(v) for v in res["mean_test_recall"]],
                        "pr_auc": [float(v) for v in res["mean_test_pr_auc"]],
                        "precision": [float(v) for v in res["mean_test_precision"]]}
    pd.DataFrame(res).drop(columns=["params"]).to_csv(f"model/cv_results_{name.replace(' ', '_')}.csv",
                                                      index=False)
    print(f"   {name}: recall_cv={T_['recall_cv']:.3f} pr_auc_cv={T_['pr_auc_cv']:.3f} "
          f"({elapsed:.0f}s) {bp}")
    return T_, rs.best_estimator_, F_, S_, A_


for name in MODELS:
    tuned[name], best[name], fold_scores[name], search_pts[name], alt[name] = cached(
        f"tuned_{name.replace(' ', '_')}", lambda n=name: run_search(n))
R["tuned"], R["fold_scores"], R["base_fold"] = tuned, fold_scores, base_fold
R["alt_criterion"] = {"name": "recall CV con precisión >= %.2f" % PREC_FLOOR, "per_model": alt,
                      "winner": max(alt, key=lambda n: (round(alt[n]["recall_cv"], 4), alt[n]["pr_auc_cv"]))}

# =================================================================== ETAPA 12: ganador por CV
ranking = sorted(tuned, key=lambda n: (tuned[n]["pr_auc_cv"], tuned[n]["recall_cv"]), reverse=True)
winner = ranking[0]
runner = ranking[1]
pipe = best[winner]
R["winner"], R["ranking_cv"] = winner, ranking
print(f"[12] Ganador por CV: {winner}")

# estabilidad de la ventaja (diferencias pareadas por fold)
stab = {}
for m in ("recall", "pr_auc"):
    d = np.array(fold_scores[winner][m]) - np.array(fold_scores[runner][m])
    stab[m] = {"vs": runner, "diffs": d.tolist(), "mean": float(d.mean()), "std": float(d.std(ddof=1)),
               "folds_won": int((d > 0).sum()), "folds_tied": int((d == 0).sum())}
R["stability"] = stab

# predicciones fuera de muestra (OOF) en Train para los 3 modelos -> umbral / curvas de validación
oof, oof_thr, oof_pr = {}, {}, {}
grid = np.linspace(0.02, 0.98, 97)


def cost_curve(yv, p):
    yv = np.asarray(yv)
    return np.array([C_FN * ((p < t) & (yv == 1)).sum() + C_FA * ((p >= t) & (yv == 0)).sum() for t in grid])


for name in tuned:
    oof[name] = cross_val_predict(clone(best[name]), X_tr, y_tr, cv=cv, method="predict_proba",
                                  params=FIT)[:, 1]
    costs = cost_curve(y_tr, oof[name])
    ties = np.where(costs == costs.min())[0]
    oof_thr[name] = float(grid[ties[len(ties) // 2]])       # centro de la meseta de costo mínimo
    p_, r_, _ = precision_recall_curve(y_tr, oof[name])
    oof_pr[name] = {"precision": p_[::max(1, len(p_) // 200)].tolist(),
                    "recall": r_[::max(1, len(r_) // 200)].tolist(),
                    "pr_auc": float(average_precision_score(y_tr, oof[name]))}
thr = oof_thr[winner]
wc = cost_curve(y_tr, oof[winner])
R["threshold"] = {"value": thr, "watch": thr / 2, "criterion": "mínimo costo esperado en OOF de Train",
                  "cost_at_thr": float(wc[np.argmin(np.abs(grid - thr))]),
                  "cost_at_0.5": float(wc[np.argmin(np.abs(grid - 0.5))]),
                  "cost_no_model": float(C_FN * y_tr.sum()), "cost_alert_all": float(C_FA * (y_tr == 0).sum()),
                  "per_model": oof_thr, "oof_pr": oof_pr}
print(f"[12] Umbral (OOF, costo mínimo, {C_FN:.0f}:{C_FA:.0f}) = {thr:.2f}")

# =================================================================== ETAPA 13: evaluación final (UNA vez)
proba_te = pipe.predict_proba(X_te)[:, 1]


def metrics_at(yv, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    return {"threshold": float(t), "recall": float(recall_score(yv, pred)),
            "precision": float(precision_score(yv, pred, zero_division=0)),
            "accuracy": float(accuracy_score(yv, pred)), "f1": float(f1_score(yv, pred)),
            "f2": float(fbeta_score(yv, pred, beta=2)), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "tp": int(tp), "cost": float(C_FN * fn + C_FA * fp),
            "roc_auc": float(roc_auc_score(yv, p)), "pr_auc": float(average_precision_score(yv, p))}


test = metrics_at(y_te, proba_te, thr)
test["at_0.5"] = metrics_at(y_te, proba_te, 0.5)
test["cost_no_model"] = float(C_FN * y_te.sum())
test["cost_alert_all"] = float(C_FA * (y_te == 0).sum())
test["cost_sensitivity"] = {}
for ratio in (5, 10, 20, 50):
    model_cost = ratio * test["fn"] + test["fp"]
    test["cost_sensitivity"][str(ratio)] = {"model": float(model_cost),
                                            "no_model": float(ratio * y_te.sum()),
                                            "alert_all": float((y_te == 0).sum())}
# errores del ganador por modo de falla (post-hoc: los modos NO se usaron como predictores)
modes_te = raw.loc[X_te.index, MODES]
pred_te = (proba_te >= thr).astype(int)
fn_mask = (y_te.values == 1) & (pred_te == 0)
test["by_mode"] = {m: {"total": int(modes_te[m][y_te.values == 1].sum()),
                       "missed": int(modes_te[m][fn_mask].sum())} for m in MODES}
test["fn_rows"] = X_te[fn_mask].round(2).head(12).to_dict("records")
test["fp_rows"] = X_te[(y_te.values == 0) & (pred_te == 1)].round(2).head(12).to_dict("records")
fpr, tpr, _ = roc_curve(y_te, proba_te)
pp, rr, _ = precision_recall_curve(y_te, proba_te)
test["roc_curve"] = {"fpr": fpr[::max(1, len(fpr) // 200)].tolist(), "tpr": tpr[::max(1, len(tpr) // 200)].tolist()}
test["pr_curve"] = {"precision": pp[::max(1, len(pp) // 200)].tolist(), "recall": rr[::max(1, len(rr) // 200)].tolist()}
R["test"] = test
print(f"[13] TEST {winner}: recall={test['recall']:.3f} precision={test['precision']:.3f} "
      f"pr_auc={test['pr_auc']:.3f} FN={test['fn']} FP={test['fp']}")

# Análisis contrafactual POST-HOC (no se usa para decidir): ¿y si se hubiera elegido mirando Test?
cf = {}
for name in tuned:
    p = best[name].predict_proba(X_te)[:, 1]
    cf[name] = metrics_at(y_te, p, oof_thr[name])
R["counterfactual"] = {
    "note": "Calculado DESPUÉS de congelar el ganador; sólo ilustra el sesgo de elegir con Test.",
    "per_model": cf, "ranking_test_pr_auc": sorted(cf, key=lambda n: -cf[n]["pr_auc"]),
    "ranking_test_recall": sorted(cf, key=lambda n: (-cf[n]["recall"], -cf[n]["pr_auc"])),
    "optimism_pr_auc": float(max(c["pr_auc"] for c in cf.values()) - cf[winner]["pr_auc"]),
}

# =================================================================== ETAPA 14: interpretación
# sobreajuste: train vs CV para la mejor config de cada modelo
gap = {}
for name in tuned:
    r = cross_validate(clone(best[name]), X_tr, y_tr, cv=cv, scoring=SCORING, params=FIT,
                       return_train_score=True)
    gap[name] = {"train_pr_auc": float(np.mean(r["train_pr_auc"])), "cv_pr_auc": float(np.mean(r["test_pr_auc"])),
                 "train_recall": float(np.mean(r["train_recall"])), "cv_recall": float(np.mean(r["test_recall"]))}
gap[winner]["test_pr_auc"], gap[winner]["test_recall"] = test["pr_auc"], test["recall"]
R["overfit"] = gap

# importancia por permutación (en Train, sin tocar Test) e importancia por impureza
pi = permutation_importance(pipe, X_tr, y_tr, scoring="average_precision", n_repeats=5,
                            random_state=SEED, n_jobs=-1)
perm = {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(X_tr.columns, pi.importances_mean, pi.importances_std)}
fnames = [n.split("__")[-1] for n in pipe.named_steps["pre"].get_feature_names_out()]
imp = dict(sorted(zip(fnames, map(float, pipe.named_steps["model"].feature_importances_)), key=lambda t: -t[1]))
R["importance"] = {"permutation": perm, "impurity": imp}

# curva de aprendizaje manual (PR-AUC) con los mismos folds
sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
lc = {s: {"train": [], "val": []} for s in sizes}
rng = np.random.default_rng(SEED)
for tr_i, va_i in cv.split(X_tr, y_tr):
    for s in sizes:
        sub = tr_i if s == 1.0 else rng.choice(tr_i, int(len(tr_i) * s), replace=False)
        yy = y_tr.iloc[sub]
        m = clone(pipe).fit(X_tr.iloc[sub], yy, model__sample_weight=compute_sample_weight("balanced", yy))
        lc[s]["train"].append(average_precision_score(yy, m.predict_proba(X_tr.iloc[sub])[:, 1]))
        lc[s]["val"].append(average_precision_score(y_tr.iloc[va_i], m.predict_proba(X_tr.iloc[va_i])[:, 1]))
R["learning_curve"] = {str(s): {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for s, d in lc.items()}

# =================================================================== ETAPA 15: cinco registros nuevos
new = pd.DataFrame([
    {"id": "R1", "descripcion": "Operación nominal, producto M, herramienta casi nueva",
     "air_temp": 299.8, "process_temp": 309.9, "rpm": 1540, "torque": 40.0, "tool_wear": 15, "type": "M"},
    {"id": "R2", "descripcion": "Herramienta muy desgastada con torque alto (riesgo de sobrecarga), tipo L",
     "air_temp": 300.4, "process_temp": 310.6, "rpm": 1420, "torque": 62.0, "tool_wear": 215, "type": "L"},
    {"id": "R3", "descripcion": "Baja velocidad y poca diferencia de temperatura (mala disipación de calor)",
     "air_temp": 301.5, "process_temp": 309.7, "rpm": 1290, "torque": 52.0, "tool_wear": 120, "type": "M"},
    {"id": "R4", "descripcion": "Torque muy bajo con velocidad alta (potencia fuera de rango)",
     "air_temp": 298.9, "process_temp": 308.7, "rpm": 2050, "torque": 12.0, "tool_wear": 60, "type": "H"},
    {"id": "R5", "descripcion": "Caso intermedio: desgaste y torque moderados-altos, tipo H",
     "air_temp": 300.9, "process_temp": 311.0, "rpm": 1480, "torque": 48.0, "tool_wear": 170, "type": "H"},
])
pn = pipe.predict_proba(new[FEATS_NUM + FEATS_CAT])[:, 1]
new["prob_falla"] = pn
new["alerta"] = np.where(pn >= thr, "ALERTA", np.where(pn >= thr / 2, "VIGILANCIA", "NORMAL"))
new["potencia_W"] = (new["torque"] * new["rpm"] * 2 * np.pi / 60).round(0)
new["dif_temp_K"] = (new["process_temp"] - new["air_temp"]).round(1)
new["desgaste_x_torque"] = (new["tool_wear"] * new["torque"]).round(0)
new.to_csv("model/nuevos_registros.csv", index=False)
R["new_records"] = new.to_dict("records")

# =================================================================== ETAPA 16: guardar
ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in FEATS_NUM}
joblib.dump({"pipeline": pipe, "threshold": thr, "watch_threshold": thr / 2, "features": FEATS_NUM + FEATS_CAT,
             "ranges": ranges, "model_name": winner, "sklearn_version": sklearn.__version__},
            "model/pipeline.joblib")
R["ranges"] = ranges
R["total_time_s"] = time.time() - t_start

# =================================================================== figuras de resultados
names = list(MODELS)
# comparación base vs optimizada
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
for a, (key, ttl) in zip(ax, [("recall", "Recall clase falla (CV)"), ("pr_auc", "PR-AUC (CV)")]):
    xs = np.arange(len(names))
    b = [base[n][key] for n in names]; bs = [base[n][f"{key}_std"] for n in names]
    t = [tuned[n][f"{key}_cv"] for n in names]; ts = [tuned[n][f"{key}_cv_std"] for n in names]
    a.bar(xs - .2, b, .4, yerr=bs, color="#bdbdbd", label="Base", capsize=3)
    a.bar(xs + .2, t, .4, yerr=ts, color=[COL[n] for n in names], label="Optimizado", capsize=3)
    a.set_xticks(xs); a.set_xticklabels([n.replace(" ", "\n") for n in names]); a.set_title(ttl); a.set_ylim(0, 1.08)
ax[0].legend(loc="lower left")
savefig("fig04_base_vs_optimizada")

# espacio de búsqueda
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for n in names:
    ax.scatter(search_pts[n]["recall"], search_pts[n]["pr_auc"], s=28, alpha=.7, color=COL[n], label=n)
    ax.scatter(tuned[n]["recall_cv"], tuned[n]["pr_auc_cv"], s=140, facecolor="none", edgecolor=COL[n], lw=2)
ax.axhline(0, c="w"); ax.set_xlabel("Recall medio CV"); ax.set_ylabel("PR-AUC medio CV")
ax.set_title(f"{N_ITER} configuraciones por modelo (círculo = elegida)"); ax.legend()
savefig("fig05_busqueda")

# PR OOF
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for n in names:
    ax.plot(oof_pr[n]["recall"], oof_pr[n]["precision"], color=COL[n], label=f"{n} (AP={oof_pr[n]['pr_auc']:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.set_title("Curvas PR en validación (OOF de Train)"); ax.legend()
savefig("fig06_pr_oof")

# costo vs umbral
fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.plot(grid, wc, color=COL[winner]); ax.axvline(thr, ls="--", c="k")
ax.text(thr + .01, wc.max() * .85, f"umbral = {thr:.2f}")
ax.set_xlabel("Umbral de decisión"); ax.set_ylabel(f"Costo (FN×{C_FN:.0f} + FP×{C_FA:.0f})")
ax.set_title(f"Selección de umbral en validación — {winner}")
savefig("fig07_costo_umbral")

# matriz de confusión
cm = np.array([[test["tn"], test["fp"]], [test["fn"], test["tp"]]])
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Falla"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Falla"])
ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Test (umbral {thr:.2f})")
savefig("fig08_confusion")

# PR y ROC test
fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
ax[0].plot(test["pr_curve"]["recall"], test["pr_curve"]["precision"], color=COL[winner])
ax[0].axhline(y_te.mean(), ls=":", c="gray"); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precisión")
ax[0].set_title(f"PR (AP={test['pr_auc']:.3f})")
ax[1].plot(test["roc_curve"]["fpr"], test["roc_curve"]["tpr"], color=COL[winner]); ax[1].plot([0, 1], [0, 1], ":", c="gray")
ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title(f"ROC (AUC={test['roc_auc']:.3f})")
savefig("fig09_pr_roc_test")

# importancias
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
pm = pd.Series({k: v["mean"] for k, v in perm.items()}).sort_values()
pe = pd.Series({k: v["std"] for k, v in perm.items()})[pm.index]
ax[0].barh(pm.index, pm.values, xerr=pe.values, color=COL[winner]); ax[0].set_title("Permutación (caída de PR-AUC)")
ims = pd.Series(imp).sort_values()
ax[1].barh(ims.index, ims.values, color="#8c8c8c"); ax[1].set_title("Importancia por impureza")
savefig("fig10_importancias")

# curva de aprendizaje
fig, ax = plt.subplots(figsize=(5, 3.4))
ns = [int(len(X_tr) * (N_FOLDS - 1) / N_FOLDS * s) for s in sizes]
for k, c, l in [("train", "#888", "Entrenamiento"), ("val", COL[winner], "Validación")]:
    m_ = np.array([R["learning_curve"][str(s)][k][0] for s in sizes]); s_ = np.array([R["learning_curve"][str(s)][k][1] for s in sizes])
    ax.plot(ns, m_, "o-", color=c, label=l); ax.fill_between(ns, m_ - s_, m_ + s_, color=c, alpha=.15)
ax.set_xlabel("Tamaño de entrenamiento"); ax.set_ylabel("PR-AUC"); ax.set_title(f"Curva de aprendizaje — {winner}"); ax.legend()
savefig("fig11_curva_aprendizaje")

# estabilidad por fold
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
for a, m, t in zip(ax, ("recall", "pr_auc"), ("Recall por fold", "PR-AUC por fold")):
    for n in names:
        a.plot(range(1, N_FOLDS + 1), fold_scores[n][m], "o-", color=COL[n], label=n)
    a.set_xticks(range(1, N_FOLDS + 1)); a.set_xlabel("Fold"); a.set_title(t)
ax[0].legend(fontsize=7)
savefig("fig12_estabilidad_folds")

# sobreajuste
fig, ax = plt.subplots(figsize=(5.6, 3.3))
xs = np.arange(len(names))
ax.bar(xs - .2, [gap[n]["train_pr_auc"] for n in names], .4, color="#bdbdbd", label="Train")
ax.bar(xs + .2, [gap[n]["cv_pr_auc"] for n in names], .4, color=[COL[n] for n in names], label="CV")
ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n") for n in names]); ax.set_ylim(0, 1.08)
ax.set_title("PR-AUC: entrenamiento vs. validación (brecha = sobreajuste)"); ax.legend(loc="lower left")
savefig("fig13_sobreajuste")

# errores por modo
fig, ax = plt.subplots(figsize=(5.2, 3.2))
tot = [test["by_mode"][m]["total"] for m in MODES]; mis = [test["by_mode"][m]["missed"] for m in MODES]
ax.bar(MODES, tot, color="#bdbdbd", label="Fallas en Test"); ax.bar(MODES, mis, color=COL["falla"], label="No detectadas")
ax.set_title("Fallas en Test por modo (un registro puede tener varios)"); ax.legend()
savefig("fig14_errores_modo")

json.dump(clean(R), open("model/results.json", "w"), indent=2, ensure_ascii=False)
print(f"Listo en {R['total_time_s'] / 60:.1f} min. Ganador: {winner}")
