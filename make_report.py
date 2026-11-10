"""Genera el informe técnico PDF a partir de model/results.json y figures/.

Uso:  python make_report.py      ->  informe/Informe_Grupo09.pdf
Se regenera completo con los números reales cada vez que se vuelve a correr train.py.
"""
import json
import math
import os
from datetime import date

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle, Preformatted)

R = json.load(open("model/results.json", encoding="utf-8"))
INFO = R["dataset"]["info"]
CFG, W, TS, TE = R["config"], R["winner"], R["tuned"], R["test"]
SYN = R["dataset"]["is_synthetic"]
MODELS = list(TS)
os.makedirs("informe", exist_ok=True)
OUT = "informe/Informe_Grupo09.pdf"
NAVY, RED, GREY = colors.HexColor("#1F3A5F"), colors.HexColor("#B3261E"), colors.HexColor("#F2F4F7")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.4,
                      alignment=TA_JUSTIFY, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, alignment=0, textColor=colors.HexColor("#444"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
                    spaceBefore=10, spaceAfter=7)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=NAVY,
                    spaceBefore=9, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, fontSize=8, spaceAfter=8)
WARN = ParagraphStyle("warn", parent=BODY, textColor=RED, fontName="Helvetica-Bold", alignment=0)

REPL = {"→": "->", "≥": ">=", "≤": "<=", "≈": "aprox.", "σ": "desv.", "−": "-", "–": "-", "—": "-",
        "“": '"', "”": '"', "’": "'", "✓": "si", "✗": "no", "Δ": "Dif. ", "·": "|"}


def T(s):
    s = str(s)
    for a, b in REPL.items():
        s = s.replace(a, b)
    return s


def P(t, st=BODY):
    return Paragraph(T(t), st)


def bullets(items, st=BODY):
    return [Paragraph(T(i), st, bulletText="\u2022") for i in items]


def table(rows, widths, header=True, font=8):
    data = [[Paragraph(T(c), CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#C5CAD3")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        st += [("BACKGROUND", (0, i), (-1, i), GREY) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return t


def fig(name, width_cm, caption):
    path = f"figures/{name}.png"
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, Paragraph(T(caption), CAP)])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


f3 = lambda v: f"{v:.3f}"
pm = lambda m, s: f"{m:.3f} +/- {s:.3f}"
S = []

# ---------------------------------------------------------------- portada
S += [Spacer(1, 3 * cm), Paragraph("GRUPO %s | TRABAJO APLICATIVO" % INFO.get("grupo", "09"),
                                   ParagraphStyle("c0", parent=BODY, alignment=TA_CENTER, textColor=NAVY, fontSize=11))]
S += [Spacer(1, .6 * cm), Paragraph("Detección de fallas en equipos industriales",
                                    ParagraphStyle("c1", parent=H1, fontSize=24, alignment=TA_CENTER, leading=28))]
S += [Paragraph("Comparación y optimización de Árbol de Decisión, Random Forest y Gradient Boosting",
                ParagraphStyle("c2", parent=BODY, alignment=TA_CENTER, fontSize=13, leading=17, textColor=colors.HexColor("#555")))]
S += [Spacer(1, 1.2 * cm), Paragraph(T(INFO.get("curso", "Sistemas Inteligentes y Machine Learning")),
                                     ParagraphStyle("c3", parent=BODY, alignment=TA_CENTER, fontSize=11))]
S += [Paragraph("Informe técnico | Clasificación binaria | Python + scikit-learn + Streamlit",
                ParagraphStyle("c4", parent=BODY, alignment=TA_CENTER, fontSize=10))]
S += [Spacer(1, 1 * cm), Paragraph("<b>Integrantes</b>", ParagraphStyle("c5", parent=BODY, alignment=TA_CENTER))]
for n in INFO.get("integrantes", []):
    S += [Paragraph(T(n), ParagraphStyle("c6", parent=BODY, alignment=TA_CENTER))]
S += [Spacer(1, .8 * cm), Paragraph(date.today().strftime("%d/%m/%Y"), ParagraphStyle("c7", parent=BODY, alignment=TA_CENTER))]
if SYN:
    S += [Spacer(1, 1.2 * cm), table([["AVISO IMPORTANTE"],
          ["Los resultados de este informe se generaron con una RÉPLICA SINTÉTICA del dataset (script "
           "generate_synthetic_data.py), creada para poder ejecutar y validar todo el sistema sin conexión. "
           "NO son los resultados del CSV real de Kaggle. Para el informe definitivo: coloque el CSV real en "
           "data/ai4i2020.csv, complete dataset_info.json y ejecute python run_all.py; este PDF se regenera solo."]],
          [15])]
S += [PageBreak()]

# ---------------------------------------------------------------- índice
S += [P("Contenido", H1)]
for i in ["1. Resumen ejecutivo", "2. Problema, objetivo y dataset", "3. Diseño experimental y métrica principal",
          "4. Desarrollo en 16 etapas (01-16)", "5. Experimentos: comparación y optimización",
          "6. Evaluación final y análisis de errores", "7. Interpretación, estabilidad y limitaciones",
          "8. Respuestas a las preguntas del trabajo", "9. Conclusiones y trabajo futuro",
          "10. Reproducibilidad y fuentes"]:
    S += [P(i)]
S += [PageBreak()]

# ---------------------------------------------------------------- 1 resumen
rk = R["ranking_cv"]
lo, hi = wilson(TE["tp"], TE["tp"] + TE["fn"])
st_pr, st_re = R["stability"]["pr_auc"], R["stability"]["recall"]
S += [P("1. Resumen ejecutivo", H1)]
if SYN:
    S += [P("Aviso: cifras obtenidas con la réplica sintética del dataset (ver portada).", WARN)]
S += [P(f"Se compararon tres algoritmos de árboles (Decision Tree, Random Forest y Gradient Boosting) para predecir falla de máquina "
        f"(1 = falla, 0 = normal) con <b>{R['dataset']['n_rows']:,}</b> observaciones, de las cuales solo <b>{R['eda']['fail_rate']:.2%}</b> son fallas. "
        f"Los tres modelos usaron la misma partición Train/Test ({R['split']['n_train']:,}/{R['split']['n_test']:,}), los mismos {CFG['n_folds']} folds, "
        f"el mismo preprocesamiento sin fuga y una búsqueda aleatoria de {CFG['n_iter']} configuraciones cada uno.")]
S += [P(f"La métrica principal, fijada antes de ajustar, fue la <b>PR-AUC</b> de la clase falla (criterio de selección) con el <b>recall</b> como métrica "
        f"operativa: el umbral de decisión se eligió en validación minimizando el costo esperado (falla no detectada = {CFG['c_fn']:.0f} unidades, "
        f"falsa alarma = {CFG['c_fa']:.0f}). Por validación cruzada ganó <b>{W}</b> (PR-AUC {pm(TS[W]['pr_auc_cv'], TS[W]['pr_auc_cv_std'])}; "
        f"segundo: {rk[1]}, {f3(TS[rk[1]]['pr_auc_cv'])}). La ventaja sobre {st_pr['vs']} fue de {st_pr['mean']:+.3f} de PR-AUC "
        f"(gana en {st_pr['folds_won']} de {CFG['n_folds']} folds).")]
S += [P(f"En Test (evaluado una sola vez) el pipeline ganador alcanzó recall <b>{f3(TE['recall'])}</b> (IC95% Wilson {lo:.2f}-{hi:.2f}), "
        f"precisión <b>{f3(TE['precision'])}</b> y PR-AUC <b>{f3(TE['pr_auc'])}</b>, con {TE['fn']} fallas no detectadas y {TE['fp']} falsas alarmas. "
        f"El modelo se guardó con joblib y se desplegó en una aplicación Streamlit de seis páginas.")]
S += [table([["Modelo", "PR-AUC CV (inicial)", "PR-AUC CV (mejor)", "Recall CV (mejor)", "Tiempo búsqueda (s)"]] +
            [[m + (" (ganador)" if m == W else ""), pm(R["base"][m]["pr_auc"], R["base"][m]["pr_auc_std"]),
              pm(TS[m]["pr_auc_cv"], TS[m]["pr_auc_cv_std"]), pm(TS[m]["recall_cv"], TS[m]["recall_cv_std"]),
              f"{TS[m]['search_time_s']:.0f}"] for m in MODELS], [3.6, 3.4, 3.4, 3.4, 3.0])]
S += [PageBreak()]

# ---------------------------------------------------------------- 2 problema y dataset
S += [P("2. Problema, objetivo y dataset", H1)]
S += [P("Etapa 01 - Definición del problema", H2)]
S += [P("Una planta industrial busca anticipar fallas mecánicas para reducir paradas no programadas y costos de mantenimiento. "
        "El problema se formula como <b>clasificación binaria supervisada</b>: dadas las condiciones de operación de una máquina "
        "(temperaturas, velocidad, torque, desgaste y tipo de producto), predecir si ocurrirá una falla (1) o no (0).")]
S += [P("<b>Utilidad de la predicción.</b> Una alerta temprana permite programar una inspección en una ventana planificada en lugar de sufrir una parada "
        "imprevista. Las dos equivocaciones tienen costos distintos: una <i>falsa alarma</i> cuesta una inspección innecesaria y una <i>falla no detectada</i> "
        f"cuesta una parada no programada. Se asumió una razón de costos {CFG['c_fn']:.0f}:{CFG['c_fa']:.0f} (supuesto ilustrativo del grupo; en planta debe "
        "reemplazarse por costos reales) y se verificó la sensibilidad con razones 5, 10, 20 y 50.")]
S += [P("Pregunta de investigación: ¿cuál de los tres algoritmos ofrece la mejor capacidad de generalización para predecir falla de máquina tras ajustar sus "
        "hiperparámetros con validación cruzada?")]
S += [P("Ficha del dataset", H2)]
rows = [["Campo", "Valor"],
        ["Nombre", INFO.get("nombre", "")], ["Autor", INFO.get("autor", "")], ["URL Kaggle", INFO.get("url_kaggle", "")],
        ["URL UCI (fuente original)", INFO.get("url_uci", "")], ["DOI", INFO.get("doi", "")],
        ["Licencia", INFO.get("licencia", "")], ["Fecha de acceso", INFO.get("fecha_acceso", "")],
        ["Observaciones | columnas", f"{R['dataset']['n_rows']:,} | {R['dataset']['n_cols']}"],
        ["Archivo usado", R["dataset"]["file"] + (" (RÉPLICA SINTÉTICA)" if SYN else "")],
        ["SHA-256 del archivo", R["dataset"]["sha256"][:32] + "..."],
        ["Variable objetivo", "Machine failure: 1 si ocurre al menos uno de los cinco modos de falla (TWF, HDF, PWF, OSF, RNF); 0 si opera normal."]]
S += [table(rows, [4.2, 12.8])]
S += [Spacer(1, 4), P("El dataset original es <b>sintético</b>: Matzka (2020) lo diseñó para reflejar el mantenimiento predictivo industrial porque los datos reales "
                      "son difíciles de obtener y publicar. Esto implica que las fallas siguen reglas explícitas (desgaste de herramienta, disipación de calor, "
                      "potencia fuera de rango, sobrecarga y fallas aleatorias) y que el desempeño no se puede extrapolar sin más a una planta real (sección 7).")]
S += [PageBreak()]

# ---------------------------------------------------------------- 3 diseño experimental
S += [P("3. Diseño experimental y métrica principal", H1)]
S += [table([["Elemento", "Decisión"],
             ["Modelos", "DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier (scikit-learn)"],
             ["Partición", f"Train {R['split']['n_train']:,} / Test {R['split']['n_test']:,} ({R['split']['test_size']:.0%}), estratificada, semilla {CFG['seed']}"],
             ["Validación cruzada", f"StratifiedKFold({CFG['n_folds']}, shuffle=True, random_state={CFG['seed']}); los MISMOS folds para los tres modelos"],
             ["Búsqueda", f"RandomizedSearchCV con {CFG['n_iter']} configuraciones aleatorias por modelo, dentro de Train"],
             ["Desbalance", "sample_weight 'balanced' idéntico para los tres modelos (los tres se entrenan con la misma ponderación)"],
             ["Métrica principal", "PR-AUC (average precision) de la clase falla, fijada ANTES del ajuste"],
             ["Métricas secundarias", "recall, precisión, F2, ROC-AUC, accuracy, matriz de confusión"],
             ["Umbral", "Elegido en validación (predicciones fuera de muestra de Train) por mínimo costo esperado; nunca en Test"],
             ["Versiones", ", ".join(f"{k} {v}" for k, v in CFG["versions"].items())]], [3.6, 13.4])]
S += [P("Justificación de la métrica principal", H2)]
S += [P("Con solo %.2f%% de fallas, la accuracy es engañosa (un modelo que nunca alerta tendría %.1f%% de accuracy). El enunciado pide priorizar "
        "recall y PR-AUC. Se adoptó la <b>PR-AUC como criterio de selección</b> porque (i) se centra en la clase rara, (ii) no depende de un umbral "
        "arbitrario, de modo que compara modelos en todos sus puntos de operación, y (iii) premia modelos que alcanzan recall alto <i>sin</i> inundar de falsas alarmas. "
        "El <b>recall</b> se prioriza en la decisión operativa: el umbral se elige minimizando FN x %.0f + FP x %.0f, lo que empuja el recall hacia arriba cuando "
        "una falla no detectada es mucho más cara. Seleccionar por recall con umbral 0.5 habría comparado modelos en puntos de operación distintos y "
        "permite 'ganar' alertando casi siempre; por eso se reporta como análisis de sensibilidad en la sección 5." % (
            R["eda"]["fail_rate"] * 100, (1 - R["eda"]["fail_rate"]) * 100, CFG["c_fn"], CFG["c_fa"]))]
S += [PageBreak()]

# ---------------------------------------------------------------- 4 etapas
S += [P("4. Desarrollo en 16 etapas", H1)]
S += [P("Cada subsección resume la evidencia mínima solicitada; el código completo está en train.py (etapas 02-16) y los gráficos en la carpeta figures/.")]

S += [P("Etapa 02 - Carga", H2)]
S += [P(f"El CSV se descargó de Kaggle y se importó con <font face='Courier'>pandas.read_csv</font>: {R['dataset']['n_rows']:,} filas y {R['dataset']['n_cols']} columnas.")]
S += [table([["Columna", "Tipo"]] + [[c, t] for c, t in R["dataset"]["dtypes"].items()], [8, 4])]

S += [P("Etapa 03 - Exploración", H2)]
e = R["eda"]
S += [P(f"Faltantes: <b>{e['missing_total']}</b>. Filas duplicadas: <b>{e['duplicated_rows']}</b>. Clases: {e['class_counts'].get('0')} normales y "
        f"{e['class_counts'].get('1')} fallas ({e['fail_rate']:.2%}). Tasa de falla por tipo de producto: " +
        ", ".join(f"{k} {v:.2%}" for k, v in e["fail_rate_by_type"].items()) + ".")]
S += [fig("fig01_clases", 12.5, "Figura 1. Balance de clases y tasa de falla por tipo de producto.")]
S += [fig("fig02_distribuciones", 14.5, "Figura 2. Distribución de cada variable según la clase (densidad). Las fallas se concentran en los extremos.")]
S += [fig("fig03_correlacion", 8.5, f"Figura 3. Correlaciones de Pearson. Velocidad y torque: r = {e['corr_rpm_torque']:.2f} (potencia casi constante).")]
S += [P("Correlaciones lineales con el objetivo: " + ", ".join(f"{k} {v:+.2f}" for k, v in e["corr_with_target"].items()) +
        ". Son débiles porque las fallas dependen de <b>relaciones no lineales y de interacciones</b> (por ejemplo, potencia = torque x velocidad, o desgaste x torque); "
        "esto justifica usar modelos basados en árboles.")]

S += [P("Etapa 04 - Limpieza inicial y revisión de fuga", H2)]
c = R["cleaning"]
S += [*bullets([
    ("<b>Fuga de información:</b> Machine failure es el O lógico de TWF, HDF, PWF, OSF y RNF" +
     (" en todas las filas" if c["target_equals_any_mode"] else
      f" salvo en {c['n_fail_without_mode'] + c['n_mode_without_fail']} filas ({c['n_fail_without_mode']} con falla pero sin modo registrado y "
      f"{c['n_mode_without_fail']} con algún modo activo pero sin falla; inconsistencias de etiquetado del propio dataset que se conservan como ruido de etiqueta)") +
     ". Esas columnas describen la causa de la falla y no existirían antes de que ocurra; se <b>excluyen</b> de X. Cuentas: " + ", ".join(f"{k}={v}" for k, v in c["mode_counts"].items()) + "."),
    f"<b>Identificadores:</b> UDI (índice 1..n) y Product ID (único por fila: {c['unique_product_ids']}) se excluyen; no tienen significado físico y permitirían memorizar filas.",
    f"<b>Type</b> coincide con la primera letra de Product ID ({c['type_matches_product_id']}), por lo que se conserva solo Type.",
    f"<b>Duplicados:</b> {e['duplicated_rows']} filas completas repetidas; {c['dup_feature_rows']} filas con predictores idénticos (se conservan: son mediciones distintas).",
    "<b>Grupos/tiempo:</b> no hay identificador de máquina repetido, por lo que no existe fuga entre registros de una misma máquina. Las temperaturas se generaron como "
    "caminata aleatoria (autocorrelación temporal); se documenta como limitación."])]

S += [P("Etapa 05 - Separar X e y", H2)]
S += [table([["Variable", "Rol", "Disponible al predecir?"],
             ["air_temp (K)", "predictor", "Sí: sensor ambiental"], ["process_temp (K)", "predictor", "Sí: sensor del proceso"],
             ["rpm", "predictor", "Sí: tacómetro"], ["torque (Nm)", "predictor", "Sí: sensor de torque"],
             ["tool_wear (min)", "predictor", "Sí: contador de uso de herramienta"], ["type (L/M/H)", "predictor", "Sí: orden de producción"],
             ["failure", "objetivo (y)", "No (es lo que se predice)"]], [4.5, 3.5, 9])]

S += [P("Etapa 06 - División Train/Test", H2)]
s_ = R["split"]
S += [P(f"División estratificada 80/20 con semilla {s_['seed']}: Train {s_['n_train']:,} filas ({s_['pos_train']} fallas, {s_['rate_train']:.2%}) y Test {s_['n_test']:,} filas "
        f"({s_['pos_test']} fallas, {s_['rate_test']:.2%}). La estratificación conserva la proporción de fallas. Test se reservó y no se usó para elegir modelos, "
        "hiperparámetros ni umbral. No se aplicó partición temporal ni por grupos porque el dataset no tiene máquinas repetidas.")]

S += [P("Etapa 07 - Preprocesamiento", H2)]
S += [*bullets([
    "<b>Imputación:</b> mediana para numéricas y moda para Type. El dataset no tiene faltantes, pero la imputación queda dentro del Pipeline para que el sistema acepte entradas incompletas en producción; sus estadísticos se aprenden solo con Train.",
    "<b>Tratamiento numérico:</b> no se escalan las variables porque los árboles son invariantes a transformaciones monótonas; omitir el escalado mantiene las unidades físicas interpretables.",
    "<b>Codificación categórica:</b> OneHotEncoder con categorías fijas [L, M, H] y handle_unknown='ignore' (no hay orden natural; evita imponer una escala falsa).",
    "<b>Sin fuga:</b> todo se ajusta con fit solo sobre Train (y dentro de cada fold durante la CV) porque el preprocesamiento vive dentro del Pipeline."])]

S += [P("Etapa 08 - Construir Pipeline", H2)]
S += [Preformatted(T("""Pipeline([
  ("pre", ColumnTransformer([
      ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), [air_temp, process_temp, rpm, torque, tool_wear]),
      ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh",  OneHotEncoder(handle_unknown="ignore", categories=[["L","M","H"]]))]), [type])])),
  ("model", <DecisionTree | RandomForest | GradientBoosting>)])"""),
                   ParagraphStyle("code", fontName="Courier", fontSize=7, leading=9, backColor=GREY, borderPadding=4))]
S += [Spacer(1, 6), P("Se construyó un Pipeline por algoritmo con idéntico preprocesamiento. Es el único objeto que se guarda y despliega.")]

S += [P("Etapa 09 - Validación cruzada base", H2)]
S += [P(f"Tres configuraciones iniciales (valores por defecto de scikit-learn; Random Forest y Gradient Boosting con 100 árboles) se evaluaron en los mismos {CFG['n_folds']} folds de Train.")]
S += [table([["Modelo", "PR-AUC", "Recall", "Precisión", "F2", "ROC-AUC"]] +
            [[m] + [pm(R["base"][m][k], R["base"][m][k + "_std"]) for k in ("pr_auc", "recall", "precision", "f2", "roc_auc")] for m in MODELS],
            [3.4, 2.7, 2.7, 2.7, 2.7, 2.8])]

S += [P("Etapa 10 - Comparación inicial", H2)]
S += [table([["Modelo", "Árboles", "Nodos totales", "Profundidad máx.", "Tiempo de ajuste/fold (s)", "PR-AUC train", "PR-AUC CV"]] +
            [[m, R["base"][m]["complexity"]["trees"], R["base"][m]["complexity"]["total_nodes"], R["base"][m]["complexity"]["max_depth"],
              f"{R['base'][m]['fit_time_s']:.2f}", f3(R["base"][m]["train_pr_auc"]), f3(R["base"][m]["pr_auc"])] for m in MODELS],
            [3.2, 1.6, 2.4, 2.4, 3.1, 2.2, 2.1])]
bm = max(MODELS, key=lambda m: R["base"][m]["pr_auc"])
S += [Spacer(1, 4), P(f"Con la configuración inicial el mejor modelo por PR-AUC es <b>{bm}</b>. Los modelos sin límite de profundidad alcanzan PR-AUC de entrenamiento muy alto "
                      "frente a la de validación: indicio de sobreajuste que la optimización debe controlar. El árbol único es el más simple y rápido; los ensambles cuestan más en tiempo y tamaño.")]

S += [P("Etapa 11 - Optimización de hiperparámetros", H2)]
S += [P(f"Se usó <b>RandomizedSearchCV</b> dentro de Train con {CFG['n_iter']} configuraciones por modelo, los mismos folds y la misma métrica (refit sobre PR-AUC). "
        "Espacios explorados (los óptimos son relativos a estos espacios):")]
S += [table([["Modelo", "Espacio de búsqueda"]] + [[m, "; ".join(f"{k}: {v}" for k, v in CFG["spaces"][m].items())] for m in MODELS], [3.2, 13.8])]
S += [Spacer(1, 5), table([["Modelo", "best_params_", "best_score_ (PR-AUC CV)", "Desv. CV", "Tiempo (s)"]] +
            [[m, ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in TS[m]["best_params"].items()), f3(TS[m]["pr_auc_cv"]),
              f3(TS[m]["pr_auc_cv_std"]), f"{TS[m]['search_time_s']:.0f}"] for m in MODELS], [2.6, 8.2, 2.3, 1.9, 2.0])]
S += [fig("fig05_busqueda", 9, "Figura 4. Las 20 configuraciones evaluadas por modelo (recall vs PR-AUC en CV). El círculo marca la elegida.")]

S += [P("Etapa 12 - Reentrenar ganador", H2)]
S += [P(f"La selección se hizo exclusivamente con la métrica de validación cruzada: ranking por PR-AUC CV = {' > '.join(rk)}. "
        f"<b>{W}</b> se reentrenó con la mejor configuración sobre todo Train (refit de RandomizedSearchCV). "
        f"Estabilidad de la ventaja frente a {st_pr['vs']} (diferencias pareadas por fold): PR-AUC {st_pr['mean']:+.3f} (desv. {st_pr['std']:.3f}, gana {st_pr['folds_won']}/{CFG['n_folds']} folds); "
        f"recall {st_re['mean']:+.3f} (desv. {st_re['std']:.3f}, gana {st_re['folds_won']}/{CFG['n_folds']} folds). Con solo {CFG['n_folds']} folds correlacionados, estas cifras son indicativas y no una prueba estadística.")]
S += [fig("fig04_base_vs_optimizada", 14, "Figura 5. Validación cruzada: configuración base vs optimizada (media y desviación estándar entre folds).")]
S += [fig("fig12_estabilidad_folds", 14, "Figura 6. Desempeño fold por fold de la mejor configuración de cada modelo.")]
th = R["threshold"]
S += [P(f"<b>Umbral de decisión.</b> Con predicciones fuera de muestra (OOF) del Train se calculó el costo FN x {CFG['c_fn']:.0f} + FP x {CFG['c_fa']:.0f} para umbrales de 0.02 a 0.98 y se tomó "
        f"el centro de la meseta de costo mínimo: <b>{th['value']:.2f}</b> (costo OOF {th['cost_at_thr']:.0f}, frente a {th['cost_at_0.5']:.0f} con umbral 0.5, "
        f"{th['cost_no_model']:.0f} sin modelo y {th['cost_alert_all']:.0f} alertando siempre).")]
S += [fig("fig07_costo_umbral", 9, "Figura 7. Costo en validación según el umbral; la línea marca el umbral elegido (sin mirar Test).")]
S += [fig("fig06_pr_oof", 9, "Figura 8. Curvas precisión-recall en validación (predicciones fuera de muestra de Train).")]
a = R["alt_criterion"]
S += [P(f"<b>Sensibilidad al criterio de selección.</b> Si en lugar de PR-AUC se hubiera seleccionado por recall CV con precisión >= {CFG['precision_floor']:.2f}, el ganador sería <b>{a['winner']}</b>:")]
S += [table([["Modelo", "Recall CV", "Precisión CV", "PR-AUC CV", "Misma config. que la elegida?"]] +
            [[m, f3(a["per_model"][m]["recall_cv"]), f3(a["per_model"][m]["precision_cv"]), f3(a["per_model"][m]["pr_auc_cv"]),
              "sí" if a["per_model"][m]["same_as_chosen"] else "no"] for m in MODELS], [3.4, 2.7, 2.9, 2.7, 5.3])]

S += [P("Etapa 13 - Evaluación final", H2)]
S += [P(f"El Pipeline ganador se evaluó <b>una sola vez</b> en Test con el umbral {TE['threshold']:.2f} fijado en validación.")]
S += [table([["Métrica", "Valor (umbral %.2f)" % TE["threshold"], "Valor (umbral 0.5)"]] +
            [[n, f3(TE[k]), f3(TE["at_0.5"][k])] for n, k in [("Recall", "recall"), ("Precisión", "precision"), ("F2", "f2"), ("F1", "f1"),
                                                              ("Accuracy", "accuracy"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]] +
            [["FN | FP | TP | TN", f"{TE['fn']} | {TE['fp']} | {TE['tp']} | {TE['tn']}", f"{TE['at_0.5']['fn']} | {TE['at_0.5']['fp']} | {TE['at_0.5']['tp']} | {TE['at_0.5']['tn']}"],
             ["Costo (FN x %.0f + FP x %.0f)" % (CFG["c_fn"], CFG["c_fa"]), f"{TE['cost']:.0f}", f"{TE['at_0.5']['cost']:.0f}"]], [6, 5.5, 5.5])]
S += [Spacer(1, 5), fig("fig08_confusion", 6.5, "Figura 9. Matriz de confusión en Test."),
      fig("fig09_pr_roc_test", 14, "Figura 10. Curvas PR y ROC en Test.")]
S += [P("Costo frente a la razón FN:FP (umbral fijo; costo de no usar modelo y de alertar siempre como referencias):")]
S += [table([["Razón FN:FP", "Costo con modelo", "Sin modelo (nunca alertar)", "Alertar siempre"]] +
            [[f"{k}:1", f"{v['model']:.0f}", f"{v['no_model']:.0f}", f"{v['alert_all']:.0f}"] for k, v in TE["cost_sensitivity"].items()], [3.5, 4.2, 5, 4.3])]
S += [Spacer(1, 5), fig("fig14_errores_modo", 9, "Figura 11. Fallas de Test por modo y cuántas no se detectaron (análisis post-hoc; los modos no son predictores).")]
mm = TE["by_mode"]
S += [P("Errores por modo (post-hoc): " + "; ".join(f"{m}: {v['missed']} de {v['total']} no detectadas" for m, v in mm.items()) +
        f". En total hubo {TE['fn']} fallas no detectadas y {TE['fp']} falsas alarmas. Ejemplos de fallas no detectadas:")]
if TE["fn_rows"]:
    cols = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "type"]
    S += [table([cols] + [[r[c] for c in cols] for r in TE["fn_rows"][:6]], [2.6, 2.9, 2.2, 2.2, 2.6, 2])]

S += [P("Etapa 14 - Interpretación", H2)]
S += [P("Los detalles están en la sección 7 (importancia de variables, sobreajuste, estabilidad y limitaciones).")]

S += [P("Etapa 15 - Predicción con cinco registros nuevos", H2)]
S += [P("Cinco configuraciones de operación plausibles y completamente descritas (valores dentro de los rangos del dataset). La alerta se emite si P(falla) >= umbral, "
        "vigilancia si P >= umbral/2.")]
nr = R["new_records"]
S += [table([["ID", "Descripción", "Aire K", "Proc. K", "rpm", "Torque", "Desg.", "Tipo", "Potencia W", "Dif. T", "P(falla)", "Estado"]] +
            [[r["id"], r["descripcion"], r["air_temp"], r["process_temp"], r["rpm"], r["torque"], r["tool_wear"], r["type"], f"{r['potencia_W']:.0f}",
              r["dif_temp_K"], f"{r['prob_falla']:.1%}", r["alerta"]] for r in nr], [.8, 4.4, 1.1, 1.2, 1, 1.1, 1.0, .9, 1.4, 1, 1.4, 1.7])]
exp = {"R1": False, "R2": True, "R3": True, "R4": True, "R5": False}
hits = [x["id"] for x in nr if (x["alerta"] == "ALERTA") == exp[x["id"]]]
miss = [x["id"] for x in nr if x["id"] not in hits]
S += [Spacer(1, 4), P("Lectura física esperada según las reglas documentadas del dataset: R2 combina desgaste alto y torque alto (desgaste x torque supera el límite de sobrecarga del tipo L); "
                      "R3 tiene diferencia de temperatura menor a 8.6 K con velocidad menor a 1380 rpm (disipación de calor deficiente); R4 tiene potencia fuera del rango 3500-9000 W; "
                      "R1 es operación nominal y R5 es un caso intermedio sin regla activada. " +
                      f"El modelo coincide con esa lectura en {len(hits)} de 5 casos ({', '.join(hits)})." +
                      (f" Discrepa en {', '.join(miss)}: el modelo no alertó un caso que la regla física marca como falla (o viceversa), lo que ilustra una limitación real: "
                       "señales como la diferencia de temperatura o la potencia son combinaciones de varias variables y un árbol sobre variables crudas las aprende de forma imperfecta. "
                       "Por eso la sección 8 propone agregar variables derivadas." if miss else " Las probabilidades son coherentes con el mecanismo físico, aunque no se le entregó explícitamente.") )]

S += [P("Etapa 16 - Guardar y desplegar", H2)]
S += [P("El Pipeline ganador se guardó con <font face='Courier'>joblib.dump</font> en model/pipeline.joblib junto con el umbral y los rangos de entrenamiento. La aplicación Streamlit (app.py) "
        "carga el artefacto una sola vez (st.cache_resource) y <b>no realiza búsqueda de hiperparámetros</b> en cada consulta. Páginas: (1) problema y dataset, (2) análisis exploratorio, "
        "(3) comparación base vs optimizada, (4) parámetros encontrados, (5) evaluación final, (6) formulario predictivo con alertas. Ejecución: <font face='Courier'>streamlit run app.py</font>.")]
S += [PageBreak()]

# ---------------------------------------------------------------- 5 experimentos
S += [P("5. Experimentos: comparación y optimización", H1)]
S += [P("Tabla resumen solicitada por el enunciado (CV inicial vs mejor CV, hiperparámetros óptimos y tiempo):")]
S += [table([["Modelo", "CV inicial (PR-AUC)", "Mejor CV (PR-AUC)", "Hiperparámetros óptimos", "Tiempo (s)"]] +
            [[m, pm(R["base"][m]["pr_auc"], R["base"][m]["pr_auc_std"]), pm(TS[m]["pr_auc_cv"], TS[m]["pr_auc_cv_std"]),
              ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in TS[m]["best_params"].items()), f"{TS[m]['search_time_s']:.0f}"] for m in MODELS],
            [2.6, 3.0, 3.0, 6.9, 1.5])]
S += [Spacer(1, 6), table([["Modelo", "Recall CV", "Precisión CV", "F2 CV", "ROC-AUC CV", "Nodos totales", "Árboles", "Predicción (ms/1000 filas)"]] +
            [[m, pm(TS[m]["recall_cv"], TS[m]["recall_cv_std"]), f3(TS[m]["precision_cv"]), f3(TS[m]["f2_cv"]), f3(TS[m]["roc_auc_cv"]),
              TS[m]["complexity"]["total_nodes"], TS[m]["complexity"]["trees"], f"{TS[m]['pred_ms_per_1000']:.1f}"] for m in MODELS],
            [2.6, 2.7, 1.9, 1.5, 2.0, 2.0, 1.5, 2.8])]
S += [Spacer(1, 6), P("<b>Por qué ganó cada configuración.</b> Con la métrica PR-AUC y 20 configuraciones por modelo, cada mejor configuración es la de mayor PR-AUC media en CV dentro del espacio explorado:")]
for m in MODELS:
    bp = TS[m]["best_params"]
    S += [Paragraph(T(f"<b>{m}:</b> " + ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in bp.items()) +
                      f". PR-AUC CV {f3(TS[m]['pr_auc_cv'])} (inicial {f3(R['base'][m]['pr_auc'])})."), BODY, bulletText="\u2022")]
S += [P("En general, los hiperparámetros que limitan la complejidad (profundidad máxima, mínimo de muestras por hoja, ccp_alpha, subsample, learning_rate bajo con más árboles) reducen la varianza y mejoran la generalización "
        "cuando hay pocas muestras positivas; los que aumentan capacidad (n_estimators, max_depth alto) ayudan hasta cierto punto y luego solo aumentan tiempo. La tabla de arriba y cv_results_*.csv permiten comprobar qué parámetros movieron realmente la métrica.")]
S += [PageBreak()]

# ---------------------------------------------------------------- 6 final
S += [P("6. Evaluación final y análisis de errores", H1)]
S += [P(f"El ganador por CV, <b>{W}</b>, obtuvo en Test: recall {f3(TE['recall'])}, precisión {f3(TE['precision'])}, F2 {f3(TE['f2'])}, PR-AUC {f3(TE['pr_auc'])} y ROC-AUC {f3(TE['roc_auc'])}. "
        f"Como Test contiene solo {TE['tp'] + TE['fn']} fallas, el recall tiene una incertidumbre apreciable: IC95% de Wilson = [{lo:.2f}, {hi:.2f}]. "
        f"Comparación Test vs validación: PR-AUC {f3(TE['pr_auc'])} frente a {f3(TS[W]['pr_auc_cv'])} en CV (diferencia {TE['pr_auc'] - TS[W]['pr_auc_cv']:+.3f}); una diferencia pequeña indica que la estimación de CV fue honesta.")]
S += [P("<b>Costo operativo.</b> Con la razón %.0f:%.0f el costo en Test fue %.0f, frente a %.0f si no se usara modelo y %.0f si se alertara siempre." % (
    CFG["c_fn"], CFG["c_fa"], TE["cost"], TE["cost_no_model"], TE["cost_alert_all"]))]
S += [P("¿Qué ocurriría si se eligiera el modelo mirando Test? (análisis post-hoc)", H2)]
cf = R["counterfactual"]
S += [P(cf["note"] + " Se evaluaron los tres modelos optimizados en Test, cada uno con su propio umbral de validación:")]
S += [table([["Modelo", "Recall", "Precisión", "PR-AUC", "FN", "FP", "Costo"]] +
            [[m, f3(cf["per_model"][m]["recall"]), f3(cf["per_model"][m]["precision"]), f3(cf["per_model"][m]["pr_auc"]),
              cf["per_model"][m]["fn"], cf["per_model"][m]["fp"], f"{cf['per_model'][m]['cost']:.0f}"] for m in MODELS], [3.6, 2, 2.2, 2.2, 1.6, 1.6, 2])]
same = cf["ranking_test_pr_auc"][0] == W
S += [Spacer(1, 4), P(f"Ranking por PR-AUC en Test: {' > '.join(cf['ranking_test_pr_auc'])}; ranking por CV: {' > '.join(rk)}. " +
                      ("Coinciden en el primer lugar. " if same else "El primer lugar cambia, lo que muestra cómo Test habría alterado la decisión. ") +
                      f"En cualquier caso, elegir mirando Test es incorrecto: con {TE['tp'] + TE['fn']} fallas el ruido de muestreo es comparable a las diferencias entre modelos, "
                      "y el máximo de varias estimaciones ruidosas está sesgado hacia arriba (optimismo de selección). Test dejaría de ser una estimación imparcial: "
                      "habría que reportar el desempeño ya optimizado sobre esos mismos datos. Por eso la decisión se tomó solo con CV y Test se usó una vez.")]
S += [PageBreak()]

# ---------------------------------------------------------------- 7 interpretación
S += [P("7. Interpretación, estabilidad y limitaciones", H1)]
S += [P("Variables importantes", H2)]
perm = R["importance"]["permutation"]
order = sorted(perm, key=lambda k: -perm[k]["mean"])
S += [P("Importancia por permutación en Train (caída de PR-AUC al barajar una variable): " + ", ".join(f"{k} {perm[k]['mean']:.3f}" for k in order) +
        f". La variable más influyente es <b>{order[0]}</b>. La importancia por impureza (derecha) puede sobrevalorar variables continuas, por eso se contrasta con la de permutación.")]
S += [fig("fig10_importancias", 14.5, "Figura 12. Importancia de variables (permutación en Train y por impureza).")]
S += [P("Sobreajuste", H2)]
ov = R["overfit"]
S += [table([["Modelo", "PR-AUC train", "PR-AUC CV", "Brecha", "Recall train", "Recall CV"]] +
            [[m, f3(ov[m]["train_pr_auc"]), f3(ov[m]["cv_pr_auc"]), f3(ov[m]["train_pr_auc"] - ov[m]["cv_pr_auc"]), f3(ov[m]["train_recall"]), f3(ov[m]["cv_recall"])] for m in MODELS],
            [3.6, 2.6, 2.6, 2.4, 2.6, 2.6])]
S += [Spacer(1, 4), fig("fig13_sobreajuste", 9, "Figura 13. PR-AUC en entrenamiento vs validación (la brecha mide sobreajuste)."),
      fig("fig11_curva_aprendizaje", 9, "Figura 14. Curva de aprendizaje del ganador: si la validación sigue subiendo, más datos ayudarían.")]
lc = R["learning_curve"]
S += [P(f"La PR-AUC de validación pasa de {lc['0.2']['val'][0]:.3f} (20% del Train) a {lc['1.0']['val'][0]:.3f} (100%); " +
        ("la curva aún crece, así que más datos probablemente mejorarían el modelo." if lc['1.0']['val'][0] - lc['0.8']['val'][0] > 0.01 else "la curva se estabiliza, así que el modelo ya aprovecha casi toda la información disponible.") +
        f" Brecha train-CV del ganador: {ov[W]['train_pr_auc'] - ov[W]['cv_pr_auc']:.3f}.")]
S += [P("Estabilidad", H2)]
S += [P(f"Entre folds, la desviación de la PR-AUC del ganador es {TS[W]['pr_auc_cv_std']:.3f} y la del recall {TS[W]['recall_cv_std']:.3f}. La ventaja sobre {st_pr['vs']} es {st_pr['mean']:+.3f} (gana en {st_pr['folds_won']} de {CFG['n_folds']} folds). "
        "Como la desviación entre folds es del mismo orden que algunas diferencias, la conclusión debe leerse como 'ventaja consistente' solo si gana en todos o casi todos los folds.")]
S += [P("Limitaciones", H2)]
S += [*bullets([
    "<b>Datos sintéticos:</b> las fallas siguen reglas explícitas; los árboles pueden aproximarlas muy bien, algo que no ocurriría con ruido de planta real.",
    f"<b>Pocas fallas:</b> {R['split']['pos_test']} positivos en Test generan intervalos de confianza amplios (recall IC95% {lo:.2f}-{hi:.2f}).",
    "<b>Autocorrelación temporal:</b> las temperaturas son caminatas aleatorias; una división aleatoria puede ser algo optimista frente a una validación cronológica.",
    "<b>Costos supuestos:</b> la razón 10:1 es un supuesto; el umbral óptimo cambia con los costos reales.",
    "<b>Probabilidades no calibradas:</b> los modelos de árboles (sobre todo con ponderación de clases) no entregan probabilidades calibradas; se usan para rankear riesgo y alertar con un umbral.",
    "<b>Horizonte:</b> el modelo predice falla en el mismo registro, no con anticipación de horas o días."])]
S += [PageBreak()]

# ---------------------------------------------------------------- 8 preguntas
S += [P("8. Respuestas a las preguntas del trabajo", H1)]
S += [P("1. ¿Qué configuración de hiperparámetros ganó en cada algoritmo y por qué?", H2)]
for m in MODELS:
    S += [Paragraph(T(f"<b>{m}:</b> " + ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in TS[m]["best_params"].items()) +
                      f". Fue la de mayor PR-AUC media en CV ({f3(TS[m]['pr_auc_cv'])} +/- {f3(TS[m]['pr_auc_cv_std'])}) entre {TS[m]['n_configs']} configuraciones evaluadas en los mismos folds."), BODY, bulletText="\u2022")]
S += [P("2. ¿Qué algoritmo ganó según validación cruzada? ¿Qué tan estable es esa ventaja?", H2)]
S += [P(f"Ganó <b>{W}</b> (PR-AUC CV {f3(TS[W]['pr_auc_cv'])}). Frente a {st_pr['vs']}: diferencia media {st_pr['mean']:+.3f}, desviación {st_pr['std']:.3f}, gana en {st_pr['folds_won']} de {CFG['n_folds']} folds (Figura 6). "
        "Con 5 folds correlacionados no se afirma significancia estadística; se informa la consistencia por fold.")]
S += [P("3. ¿Qué errores y limitaciones mostró el ganador en el conjunto Test?", H2)]
S += [P(f"{TE['fn']} fallas no detectadas ({', '.join(f'{m}: {v['missed']}/{v['total']}' for m, v in mm.items() if v['total'])} por modo) y {TE['fp']} falsas alarmas; precisión {f3(TE['precision'])}. "
        f"Limitaciones: pocas fallas en Test (IC95% de recall {lo:.2f}-{hi:.2f}), datos sintéticos y costos supuestos (sección 7).")]
S += [P("4. ¿Qué ocurriría si se eligiera el modelo mirando Test?", H2)]
S += [P("Test dejaría de ser independiente: la elección se ajustaría al ruido de esos datos y la métrica reportada sería optimista (sesgo de selección). "
        f"En este trabajo, el ranking por Test fue {' > '.join(cf['ranking_test_pr_auc'])} y por CV {' > '.join(rk)} (sección 6).")]
S += [P("5. ¿Qué cambiaría antes de utilizarlo en un entorno real?", H2)]
S += [*bullets([
    "Validar con datos reales de planta y con partición cronológica o por máquina.",
    "Estimar los costos reales de falsa alarma y de parada no programada y recalcular el umbral.",
    "Calibrar probabilidades (Platt/isotónica) y fijar niveles de alerta con operación y mantenimiento.",
    "Agregar variables físicas derivadas (potencia = torque x velocidad, diferencia de temperatura, desgaste x torque) y evaluar su aporte con la misma CV.",
    "Monitorear deriva de datos y rendimiento, y reentrenar periódicamente; mantener un esquema con humano en el lazo.",
    "Definir el horizonte de predicción (falla en las próximas horas) y rediseñar la variable objetivo en consecuencia.",
    "Pruebas de robustez: valores faltantes, sensores descalibrados, rangos fuera de entrenamiento (la app ya avisa)."])]

# ---------------------------------------------------------------- 9 conclusiones
S += [P("9. Conclusiones y trabajo futuro", H1)]
S += [*bullets([
    f"Bajo el mismo problema, partición y folds, <b>{W}</b> fue el mejor por validación cruzada (PR-AUC {f3(TS[W]['pr_auc_cv'])}) y su desempeño en Test ({f3(TE['pr_auc'])} PR-AUC; recall {f3(TE['recall'])}) fue consistente con esa estimación.",
    "Optimizar hiperparámetros con CV dentro de Train mejora o iguala la configuración inicial y controla el sobreajuste; los resultados óptimos son relativos al espacio explorado.",
    "La elección del umbral por costo en validación convierte las probabilidades en una regla de alerta alineada con el costo de negocio, sin tocar Test.",
    "El análisis de sensibilidad muestra que la conclusión depende de la métrica: seleccionar por recall con umbral fijo favorece modelos que alertan en exceso. Fijar la métrica antes del ajuste y justificarla es parte del resultado.",
    "Trabajo futuro: datos reales, validación temporal, calibración, variables derivadas y comparación con otros modelos."])]

# ---------------------------------------------------------------- 10 reproducibilidad y fuentes
S += [P("10. Reproducibilidad y fuentes", H1)]
S += [P("Reproducibilidad", H2)]
S += [Preformatted(T("""pip install -r requirements.txt
# 1) colocar el CSV real en data/ai4i2020.csv  (Kaggle/UCI, ver README) y completar dataset_info.json
python run_all.py          # entrena, genera figuras, informe PDF y presentación
streamlit run app.py       # aplicación de predicción"""), ParagraphStyle("code2", fontName="Courier", fontSize=7.5, leading=10, backColor=GREY, borderPadding=4))]
S += [Spacer(1, 6), P(f"Semilla {CFG['seed']}; scikit-learn {CFG['versions']['sklearn']}; pandas {CFG['versions']['pandas']}; numpy {CFG['versions']['numpy']}; Python {CFG['versions']['python']}. "
                      f"Presupuesto de búsqueda: {CFG['n_iter']} configuraciones x {CFG['n_folds']} folds x 3 modelos. Tiempo total de la ejecución: {R['total_time_s'] / 60:.1f} min.")]
S += [P("Fuentes", H2)]
S += [*bullets([
    T(INFO.get("cita", "")),
    "AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C (licencia CC BY 4.0).",
    "Kaggle: " + INFO.get("url_kaggle", ""),
    "Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830, 2011.",
    "Bergstra, J. y Bengio, Y. Random Search for Hyper-Parameter Optimization. JMLR 13, 281-305, 2012.",
    "Saito, T. y Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 2015.",
    "Guía del trabajo aplicativo Grupo 09, Sistemas Inteligentes y Machine Learning."], SMALL)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.1 * cm, "Sistemas Inteligentes y Machine Learning | Trabajo aplicativo grupal | Grupo 09")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {doc.page}")
    if SYN:
        canvas.setFillColor(RED)
        canvas.drawString(2 * cm, A4[1] - 1.1 * cm, "Resultados con réplica sintética del dataset (no es el CSV real)")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Detección de fallas en equipos industriales - Grupo 09", author="Grupo 09")
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Informe generado:", OUT)
