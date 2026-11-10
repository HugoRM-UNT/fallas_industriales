"""App Streamlit - Detección de fallas en equipos industriales (Grupo 09).

Ejecutar:  streamlit run app.py
La página "0. Entrenamiento" lanza train.py (búsqueda de hiperparámetros) SOLO cuando se pulsa el botón.
Las demás páginas, incl. el formulario predictivo, únicamente cargan el Pipeline ganador (joblib).
"""
import json
import os
import shutil
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fallas industriales - Grupo 09", page_icon="🛠️", layout="wide")

HAS_MODEL = os.path.exists("model/results.json") and os.path.exists("model/pipeline.joblib")


@st.cache_resource
def load_model():
    return joblib.load("model/pipeline.joblib")


@st.cache_data
def load_results():
    return json.load(open("model/results.json", encoding="utf-8"))


@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


if HAS_MODEL:
    R, B = load_results(), load_model()
    PIPE, THR, WATCH = B["pipeline"], B["threshold"], B["watch_threshold"]
    WIN = R["winner"]
    MODELS = list(R["tuned"])


def fig(name, caption=None):
    p = f"figures/{name}.png"
    if os.path.exists(p):
        st.image(p, caption=caption)


PROGRESS = {"[02]": 5, "[04]": 10, "[09-10]": 15, "[11]": 30, "Decision Tree: recall_cv": 40,
            "Random Forest: recall_cv": 60, "Gradient Boosting: recall_cv": 78, "[12]": 82, "[13]": 92, "Listo": 100}


def training_page():
    st.title("Entrenamiento de modelos")
    st.caption("Ejecuta train.py (etapas 02-16): EDA, split, Pipelines, CV base, RandomizedSearchCV, ganador, "
               "umbral, evaluación en Test, importancias y guardado con joblib.")
    if HAS_MODEL:
        st.info(f"Modelo actual: **{WIN}** · entrenado con {R['config']['n_iter']} configuraciones por modelo "
                f"en {R['total_time_s'] / 60:.1f} min · archivo {R['dataset']['file']}")
    else:
        st.warning("Todavía no hay modelo entrenado. Pulsa «Iniciar entrenamiento».")
    c1, c2 = st.columns(2)
    n_iter = c1.slider("Configuraciones aleatorias por modelo (n_iter)", 2, 40, 20,
                       help="20 = presupuesto sugerido por el enunciado. 5 = prueba rápida.")
    fresh = c2.checkbox("Reentrenar desde cero (borrar checkpoints)", value=False,
                        help="Si no se marca, se reutilizan resultados guardados con los mismos datos y n_iter.")
    up = c1.file_uploader("CSV propio (opcional; por defecto data/ai4i2020.csv)", type="csv")
    if st.button("Iniciar entrenamiento", type="primary"):
        data = "data/ai4i2020.csv"
        if up is not None:
            data = "data/subido.csv"
            os.makedirs("data", exist_ok=True)
            open(data, "wb").write(up.getbuffer())
        if not os.path.exists(data):
            st.error(f"No existe {data}.")
            return
        if fresh:
            shutil.rmtree("model/cache", ignore_errors=True)
        cmds = [[sys.executable, "-u", "train.py", "--n-iter", str(n_iter), "--data", data]]
        # if docs:
        #    cmds += [[sys.executable, "make_report.py"], [sys.executable, "make_slides.py"]]
        bar, box, log, ok = st.progress(0, text="Iniciando..."), st.empty(), [], True
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with st.status("Entrenando (puede tardar varios minutos)...", expanded=True) as status:
            for cmd in cmds:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", env=env)
                for line in proc.stdout:
                    log.append(line.rstrip())
                    box.code("\n".join(log[-20:]))
                    for k, v in PROGRESS.items():
                        if k in line:
                            bar.progress(v, text=line.strip()[:90])
                if proc.wait() != 0:
                    ok = False
                    break
            status.update(label="Entrenamiento terminado" if ok else "Error durante el entrenamiento",
                          state="complete" if ok else "error")
        if ok:
            bar.progress(100, text="Listo")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Modelo entrenado y guardado. Abre las otras páginas para ver los resultados "
                       "(si no se actualizan, recarga la página con F5).")
        else:
            st.error("El entrenamiento falló. Revisa el registro de arriba.")


st.sidebar.title("🛠️ Grupo 09")
st.sidebar.caption("Detección de fallas en equipos industriales")
PAGES = ["0. Entrenamiento", "1. Problema y dataset", "2. Análisis exploratorio",
         "3. Base vs. optimizada", "4. Parámetros encontrados",
         "5. Evaluación final", "6. Formulario predictivo"]
page = st.sidebar.radio("Página", PAGES if HAS_MODEL else PAGES[:1])
if HAS_MODEL:
    st.sidebar.markdown(f"**Modelo ganador:** {WIN}  \n**Umbral:** {THR:.2f}")
    if R["dataset"]["is_synthetic"]:
        st.sidebar.warning("Datos: RÉPLICA SINTÉTICA (no es el CSV real de Kaggle).")

# --------------------------------------------------------------------------- 0
if page.startswith("0"):
    training_page()

# --------------------------------------------------------------------------- 1
elif page.startswith("1"):
    st.title("Detección de fallas en equipos industriales")
    if R["dataset"]["is_synthetic"]:
        st.warning("Esta ejecución usa una réplica sintética del dataset para poder probar el sistema. "
                   "Reemplace data/ai4i2020.csv por el CSV real de Kaggle y vuelva a ejecutar train.py.")
    st.markdown("""
**Objetivo.** Predecir si una máquina fallará (**1 = falla, 0 = normal**) a partir de condiciones de operación,
para programar mantenimiento antes de una parada no programada.

**Pregunta de investigación.** ¿Cuál de los tres algoritmos (Árbol de Decisión, Random Forest, Gradient Boosting)
generaliza mejor tras ajustar sus hiperparámetros con validación cruzada?
""")
    d, s, info = R["dataset"], R["split"], R["dataset"]["info"]
    c = st.columns(5)
    c[0].metric("Observaciones", f"{d['n_rows']:,}")
    c[1].metric("Train / Test", f"{s['n_train']:,} / {s['n_test']:,}")
    c[2].metric("% fallas", f"{R['eda']['fail_rate']:.2%}")
    c[3].metric("Predictores", 6)
    c[4].metric("Faltantes", R["eda"]["missing_total"])
    st.subheader("Ficha del dataset")
    st.table(pd.Series({"Nombre": info.get("nombre"), "Autor": info.get("autor"),
                        "URL Kaggle": info.get("url_kaggle"), "URL UCI": info.get("url_uci"),
                        "Licencia": info.get("licencia"), "Fecha de acceso": info.get("fecha_acceso")},
                       name="Valor").astype(str))
    st.subheader("Variables")
    st.markdown("""
| Variable | Rol | Descripción |
|---|---|---|
| Air temperature [K] | predictor | temperatura del aire |
| Process temperature [K] | predictor | temperatura del proceso |
| Rotational speed [rpm] | predictor | velocidad de rotación |
| Torque [Nm] | predictor | torque |
| Tool wear [min] | predictor | desgaste acumulado de la herramienta |
| Type (L/M/H) | predictor | calidad del producto |
| Machine failure | **objetivo** | 1 si ocurre al menos un modo de falla |
| TWF, HDF, PWF, OSF, RNF | **excluidas** | causas de la falla → fuga de información |
| UDI, Product ID | **excluidas** | identificadores |
""")
    st.caption(f"Semilla {R['config']['seed']} · scikit-learn {R['config']['versions']['sklearn']} · "
               f"{R['config']['n_iter']} configuraciones aleatorias por modelo · {R['config']['n_folds']} folds")

# --------------------------------------------------------------------------- 2
elif page.startswith("2"):
    st.title("Análisis exploratorio")
    e = R["eda"]
    c = st.columns(4)
    c[0].metric("Faltantes", e["missing_total"]); c[1].metric("Filas duplicadas", e["duplicated_rows"])
    c[2].metric("Fallas", e["class_counts"].get("1")); c[3].metric("Normales", e["class_counts"].get("0"))
    fig("fig01_clases"); fig("fig02_distribuciones")
    l, r = st.columns(2)
    with l:
        fig("fig03_correlacion")
    with r:
        st.markdown("**Lectura rápida**")
        st.markdown(f"- Desbalance fuerte: {e['fail_rate']:.2%} de fallas → la *accuracy* engaña; se prioriza PR-AUC y recall.\n"
                    f"- Velocidad y torque están correlacionados (r = {e['corr_rpm_torque']:.2f}) porque la potencia es ≈ constante.\n"
                    "- Las fallas aparecen en los extremos (desgaste alto, torque alto/bajo, baja velocidad), "
                    "es decir, relaciones **no lineales** → modelos de árboles.")
        st.info("Las columnas TWF/HDF/PWF/OSF/RNF codifican la causa de la falla: el objetivo es (casi siempre) su 'O' lógico, "
                f"con {R['cleaning'].get('n_fail_without_mode', 0) + R['cleaning'].get('n_mode_without_fail', 0)} filas inconsistentes; "
                "por eso se excluyen (fuga de información).")
    df = load_data(R["dataset"]["file"])
    if df is not None:
        st.subheader("Explorador interactivo")
        col = st.selectbox("Variable", ["Air temperature [K]", "Process temperature [K]",
                                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
        g = (df.assign(bin=pd.cut(df[col], 25)).groupby(["bin", "Machine failure"], observed=True)
             .size().unstack(fill_value=0).rename(columns={0: "Normal", 1: "Falla"}))
        g.index = [round(i.mid, 1) for i in g.index]
        st.bar_chart(g)
        st.dataframe(df.head(20))

# --------------------------------------------------------------------------- 3
elif page.startswith("3"):
    st.title("Validación cruzada: configuración base vs. optimizada")
    st.caption(f"Mismos {R['config']['n_folds']} folds de Train y misma métrica principal: {R['config']['primary_metric']}.")
    rows = []
    for m in MODELS:
        b, t = R["base"][m], R["tuned"][m]
        rows.append({"Modelo": m,
                     "PR-AUC base": f"{b['pr_auc']:.3f} ± {b['pr_auc_std']:.3f}",
                     "PR-AUC optimizada": f"{t['pr_auc_cv']:.3f} ± {t['pr_auc_cv_std']:.3f}",
                     "Recall base": f"{b['recall']:.3f} ± {b['recall_std']:.3f}",
                     "Recall optimizada": f"{t['recall_cv']:.3f} ± {t['recall_cv_std']:.3f}",
                     "Tiempo búsqueda (s)": round(t["search_time_s"], 1)})
    st.dataframe(pd.DataFrame(rows))
    st.success(f"Ganador por CV: **{WIN}**")
    fig("fig04_base_vs_optimizada")
    l, r = st.columns(2)
    with l:
        fig("fig05_busqueda")
    with r:
        fig("fig12_estabilidad_folds")
    s = R["stability"]["pr_auc"]
    st.markdown(f"**Estabilidad de la ventaja** ({WIN} vs {s['vs']}): diferencia media de PR-AUC = "
                f"{s['mean']:+.3f} (σ = {s['std']:.3f}); gana en {s['folds_won']} de {R['config']['n_folds']} folds.")
    st.subheader("Complejidad y tiempos (configuración base → optimizada)")
    st.dataframe(pd.DataFrame([{"Modelo": m,
                                "Nodos base": R["base"][m]["complexity"]["total_nodes"],
                                "Nodos opt.": R["tuned"][m]["complexity"]["total_nodes"],
                                "Árboles opt.": R["tuned"][m]["complexity"]["trees"],
                                "Fit base (s)": round(R["base"][m]["fit_time_s"], 2),
                                "Fit opt. (s)": round(R["tuned"][m]["mean_fit_time_s"], 2),
                                "Predicción (ms/1000)": round(R["tuned"][m]["pred_ms_per_1000"], 1)} for m in MODELS]))

# --------------------------------------------------------------------------- 4
elif page.startswith("4"):
    st.title("Hiperparámetros óptimos")
    for m in MODELS:
        t = R["tuned"][m]
        with st.expander(f"{m} {'🏆' if m == WIN else ''}", expanded=(m == WIN)):
            st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t["best_params"].items()})
            st.write(f"best_score_ (PR-AUC CV) = **{t['pr_auc_cv']:.4f}** ± {t['pr_auc_cv_std']:.4f} · "
                     f"recall CV = {t['recall_cv']:.3f} · precisión CV = {t['precision_cv']:.3f} · "
                     f"{t['n_configs']} configuraciones · {t['search_time_s']:.0f} s")
    st.subheader("Espacio de búsqueda explorado")
    st.json(R["config"]["spaces"])
    st.subheader("Sensibilidad al criterio de selección")
    a = R["alt_criterion"]
    st.write(f"Criterio alternativo: **{a['name']}** → ganador {a['winner']}.")
    st.dataframe(pd.DataFrame(a["per_model"]).T[["recall_cv", "precision_cv", "pr_auc_cv"]].round(3))

# --------------------------------------------------------------------------- 5
elif page.startswith("5"):
    t = R["test"]
    st.title(f"Evaluación final en Test — {WIN}")
    st.caption("Test se evaluó UNA sola vez, con el Pipeline ganador y el umbral fijado en validación.")
    c = st.columns(6)
    for col, (k, n) in zip(c, [("recall", "Recall"), ("precision", "Precisión"), ("f2", "F2"),
                               ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy")]):
        col.metric(n, f"{t[k]:.3f}")
    l, r = st.columns(2)
    with l:
        fig("fig08_confusion")
    with r:
        fig("fig09_pr_roc_test")
    st.markdown(f"**Errores:** {t['fn']} fallas no detectadas (FN) y {t['fp']} falsas alarmas (FP). "
                f"Umbral = {t['threshold']:.2f} (mínimo costo en validación, FN:FP = {R['config']['c_fn']:.0f}:{R['config']['c_fa']:.0f}).")
    st.subheader("Costo: falsas alarmas vs. fallas no detectadas")
    cs = pd.DataFrame(t["cost_sensitivity"]).T
    cs.index.name = "Costo FN : FP"
    cs.columns = ["Modelo", "Sin modelo (nunca alertar)", "Alertar siempre"]
    st.dataframe(cs.round(0))
    l, r = st.columns(2)
    with l:
        fig("fig14_errores_modo")
    with r:
        fig("fig10_importancias")
    l, r = st.columns(2)
    with l:
        fig("fig11_curva_aprendizaje")
    with r:
        fig("fig13_sobreajuste")
    st.subheader("¿Qué pasaría si se eligiera mirando Test? (análisis post-hoc)")
    cf = R["counterfactual"]
    st.caption(cf["note"])
    st.dataframe(pd.DataFrame(cf["per_model"]).T[["recall", "precision", "pr_auc", "cost"]].round(3))

# --------------------------------------------------------------------------- 6
else:
    st.title("Formulario predictivo")
    st.caption(f"Modelo: {WIN} · alerta si P(falla) ≥ {THR:.2f} · vigilancia si ≥ {WATCH:.2f}")
    rg = B["ranges"]

    def verdict(p):
        if p >= THR:
            return "🔴 ALERTA: programar inspección de mantenimiento"
        if p >= WATCH:
            return "🟡 VIGILANCIA: monitorear de cerca el equipo"
        return "🟢 NORMAL: operación sin señales de falla"

    tab1, tab2 = st.tabs(["Consulta individual", "Cinco registros de prueba"])
    with tab1:
        a, b, c = st.columns(3)
        air = a.number_input("Temperatura del aire [K]", 290.0, 310.0, 300.0, 0.1)
        proc = b.number_input("Temperatura del proceso [K]", 300.0, 320.0, 310.0, 0.1)
        rpm = c.number_input("Velocidad de rotación [rpm]", 500, 4000, 1500, 10)
        tq = a.number_input("Torque [Nm]", 0.5, 100.0, 40.0, 0.5)
        wear = b.number_input("Desgaste de herramienta [min]", 0, 400, 100, 1)
        typ = c.selectbox("Tipo de producto", ["L", "M", "H"])
        x = pd.DataFrame([{"air_temp": air, "process_temp": proc, "rpm": rpm, "torque": tq,
                           "tool_wear": wear, "type": typ}])
        out = [k for k, v in {"air_temp": air, "process_temp": proc, "rpm": rpm, "torque": tq, "tool_wear": wear}.items()
               if not (rg[k][0] * .9 <= v <= rg[k][1] * 1.1)]
        if out:
            st.warning("Valores fuera del rango visto en entrenamiento (" + ", ".join(out) + "): la predicción es una extrapolación.")
        if st.button("Predecir", type="primary"):
            p = float(PIPE.predict_proba(x)[0, 1])
            st.metric("Probabilidad de falla", f"{p:.1%}")
            st.progress(min(1.0, p))
            st.markdown(f"### {verdict(p)}")
            st.caption(f"Potencia = {tq * rpm * 2 * 3.14159265 / 60:,.0f} W · ΔT = {proc - air:.1f} K · "
                       f"desgaste × torque = {wear * tq:,.0f}")
    with tab2:
        st.write("Cinco configuraciones de operación completamente descritas (etapa 15):")
        nr = pd.DataFrame(R["new_records"])
        nr["prob_falla"] = nr["prob_falla"].map("{:.1%}".format)
        st.dataframe(nr[["id", "descripcion", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
                         "type", "potencia_W", "dif_temp_K", "desgaste_x_torque", "prob_falla", "alerta"]])
