"""Genera presentacion/Presentacion_Grupo09.pptx (15 diapositivas) desde model/results.json."""
import json, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

with open("model/results.json", encoding="utf-8") as results_file:
    R = json.load(results_file)
INFO, CFG, W, TS, TE = R["dataset"]["info"], R["config"], R["winner"], R["tuned"], R["test"]
SYN, MODELS, RK = R["dataset"]["is_synthetic"], list(R["tuned"]), R["ranking_cv"]
NAVY, RED, GREY = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0xB3, 0x26, 0x1E), RGBColor(0x55, 0x55, 0x55)
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BL = prs.slide_layouts[6]
f3 = lambda v: f"{v:.3f}"


def tb(s, x, y, w, h, text, size=16, bold=False, color=None, align=None):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.word_wrap = True
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.bold = bold; p.space_after = Pt(6)
        if color: p.font.color.rgb = color
        if align: p.alignment = align
    return box


def slide(title, notes=""):
    s = prs.slides.add_slide(BL)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb(s, .6, .3, 12, .9, title, 30, True, NAVY)
    if SYN: tb(s, .6, 7.05, 9, .3, "Cifras con réplica sintética del dataset (no es el CSV real)", 10, False, RED)
    tb(s, 11.2, 7.05, 2, .3, f"Grupo 09 | {len(prs.slides)}", 10, False, GREY)
    s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(s, items, x=.6, y=1.5, w=12, h=5, size=18):
    tb(s, x, y, w, h, ["• " + i for i in items], size)


def pic(s, name, x, y, w):
    s.shapes.add_picture(f"figures/{name}.png", Inches(x), Inches(y), width=Inches(w))


def tbl(s, rows, x, y, w, size=12, colw=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(.4 * len(rows))).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs: p.font.size = Pt(size); p.font.bold = (i == 0)
    if colw:
        for j, cw in enumerate(colw): t.columns[j].width = Inches(cw)


# 1 portada
s = prs.slides.add_slide(BL)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
W_ = RGBColor(255, 255, 255)
tb(s, .8, 2, 11.5, 1.5, "Detección de fallas en equipos industriales", 42, True, W_)
tb(s, .8, 3.5, 11.5, 1, "Árbol de Decisión vs Random Forest vs Gradient Boosting: optimización con validación cruzada", 22, False, W_)
tb(s, .8, 5, 11.5, 1.5, ["Grupo 09 | " + INFO.get("curso", ""), ", ".join(INFO.get("integrantes", []))], 16, False, W_)
s.notes_slide.notes_text_frame.text = "Presentación del grupo y tema. Cada integrante se presenta."

s = slide("1. Problema y objetivo", "Utilidad: anticipar paradas no programadas. Costo asimétrico FN vs FP.")
bullets(s, ["Planta industrial: reducir paradas no programadas y costos de mantenimiento.",
            "Clasificación binaria: falla (1) vs normal (0) a partir de condiciones de operación.",
            f"Falla no detectada cuesta {CFG['c_fn']:.0f}x una falsa alarma (supuesto ilustrativo).",
            "Pregunta: ¿qué algoritmo generaliza mejor tras optimizar hiperparámetros con CV?"])

s = slide("2. Dataset: AI4I 2020 (Matzka)", "URL, licencia, fecha y significado del objetivo. Dataset sintético modelado sobre una fresadora.")
d = R["dataset"]
bullets(s, [f"{d['n_rows']:,} observaciones, {d['n_cols']} columnas; fallas = {R['eda']['fail_rate']:.2%}.",
            "Predictores: temp. aire, temp. proceso, rpm, torque, desgaste de herramienta, tipo (L/M/H).",
            "Objetivo: Machine failure = 1 si ocurre algún modo (TWF, HDF, PWF, OSF, RNF).",
            "Excluidos por fuga: TWF/HDF/PWF/OSF/RNF (causa de la falla), UDI y Product ID.",
            f"Fuente: {INFO.get('url_uci','')} | Licencia: CC BY 4.0 (UCI). Acceso: {INFO.get('fecha_acceso','')}"], size=17)

s = slide("3. Metodología", "Mismo split y mismos folds para los tres modelos. Test intocable.")
tbl(s, [["Elemento", "Decisión"], ["Split", f"80/20 estratificado, semilla {CFG['seed']}"], ["CV", f"StratifiedKFold {CFG['n_folds']} folds, iguales para los 3 modelos"],
        ["Búsqueda", f"RandomizedSearchCV, {CFG['n_iter']} configs por modelo, solo en Train"], ["Métrica principal", "PR-AUC de la clase falla (fijada antes)"],
        ["Operativa", "Recall con umbral de mínimo costo en validación"], ["Desbalance", "sample_weight balanceado, igual en los 3"]], .6, 1.6, 12.1, 15, [3, 9.1])

s = slide("4. Exploración", "Desbalance fuerte; relaciones no lineales; fallas en extremos.")
pic(s, "fig01_clases", .5, 1.4, 6); pic(s, "fig02_distribuciones", 6.7, 1.4, 6.2)
tb(s, .6, 4.6, 6, 2, [f"Faltantes: {R['eda']['missing_total']} | Duplicados: {R['eda']['duplicated_rows']}", "Accuracy engaña: priorizar PR-AUC y recall."], 16)

s = slide("5. Preprocesamiento y Pipeline", "Cada integrante explica una decisión: imputación, sin escalado, OneHot, sin fuga.")
bullets(s, ["Imputación: mediana (numéricas) y moda (Type), aprendida solo con Train.",
            "Sin escalado: los árboles son invariantes a escala; se conservan unidades físicas.",
            "OneHotEncoder fijo [L, M, H], handle_unknown='ignore'.",
            "ColumnTransformer + estimador dentro de un Pipeline por algoritmo: sin fuga en CV.",
            "Fuga revisada: el objetivo es (casi) el O lógico de los 5 modos de falla -> se excluyen."])

s = slide("6. Espacios de hiperparámetros", "Explicar un hiperparámetro por integrante: max_depth, min_samples_leaf, learning_rate, subsample...")
tbl(s, [["Modelo", "Hiperparámetros explorados"]] + [[m, ", ".join(CFG["spaces"][m].keys())] for m in MODELS], .6, 1.6, 12.1, 14, [3, 9.1])
tb(s, .6, 4.2, 12, 1, f"Presupuesto: {CFG['n_iter']} configs x {CFG['n_folds']} folds x 3 modelos. Óptimos relativos al espacio explorado.", 16)

s = slide("7. CV: base vs. optimizada", "La optimización mejora o iguala la base y controla el sobreajuste.")
pic(s, "fig04_base_vs_optimizada", .5, 1.4, 7.8)
tbl(s, [["Modelo", "PR-AUC base", "PR-AUC opt."]] + [[m, f3(R["base"][m]["pr_auc"]), f3(TS[m]["pr_auc_cv"])] for m in MODELS], 8.5, 1.6, 4.4, 12)

s = slide(f"8. Ganador por CV: {W}", "Lógica de selección: mejor PR-AUC media en CV, mismos folds, sin Test.")
st = R["stability"]["pr_auc"]
bullets(s, [f"Ranking CV: {' > '.join(RK)}.", f"Ventaja sobre {st['vs']}: {st['mean']:+.3f} PR-AUC; gana {st['folds_won']}/{CFG['n_folds']} folds.",
            f"Mejores parámetros: " + ", ".join(f"{k}={round(v,3) if isinstance(v,float) else v}" for k, v in TS[W]["best_params"].items())], 1, 1.4, 5.2, 5, 15)
pic(s, "fig12_estabilidad_folds", 6.2, 1.5, 6.8)

s = slide("9. Umbral por costo (en validación)", "El umbral se eligió con predicciones OOF de Train, nunca con Test.")
pic(s, "fig07_costo_umbral", .5, 1.4, 6); pic(s, "fig06_pr_oof", 6.8, 1.4, 6)
tb(s, .6, 5.3, 12, 1, f"Umbral = {R['threshold']['value']:.2f} (FN x {CFG['c_fn']:.0f} + FP x {CFG['c_fa']:.0f}).", 18)

s = slide("10. Evaluación final en Test (una sola vez)", "Test evaluado una vez con el pipeline ganador.")
pic(s, "fig08_confusion", .5, 1.4, 4.6)
tbl(s, [["Métrica", "Valor"]] + [[n, f3(TE[k])] for n, k in [("Recall", "recall"), ("Precisión", "precision"), ("F2", "f2"), ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc")]] +
    [["FN | FP", f"{TE['fn']} | {TE['fp']}"]], 5.6, 1.6, 4, 14)
pic(s, "fig09_pr_roc_test", 9.8, 1.6, 3.4)

s = slide("11. Interpretación y sobreajuste", "Importancia por permutación; brecha train-CV; curva de aprendizaje.")
pic(s, "fig10_importancias", .4, 1.4, 6.6); pic(s, "fig13_sobreajuste", 7.1, 1.4, 5.8)
pic(s, "fig11_curva_aprendizaje", 3.4, 4.6, 3.4)

s = slide("12. ¿Y si se eligiera mirando Test?", "Optimismo de selección: Test deja de ser imparcial.")
cf = R["counterfactual"]
tbl(s, [["Modelo", "PR-AUC Test", "Recall", "Costo"]] + [[m, f3(cf["per_model"][m]["pr_auc"]), f3(cf["per_model"][m]["recall"]), f"{cf['per_model'][m]['cost']:.0f}"] for m in MODELS], .6, 1.6, 8, 14)
bullets(s, ["Análisis post-hoc, solo ilustrativo.", f"Ranking Test: {' > '.join(cf['ranking_test_pr_auc'])}.", f"Ranking CV: {' > '.join(RK)}.",
            "Elegir con Test sesga la métrica hacia arriba (pocas fallas = ruido alto)."], .6, 3.9, 12, 3, 16)

s = slide("13. Predicción de 5 registros y app Streamlit", "Mostrar la app en vivo: formulario y alertas.")
tbl(s, [["ID", "Descripción", "P(falla)", "Estado"]] + [[r["id"], r["descripcion"][:70], f"{r['prob_falla']:.1%}", r["alerta"]] for r in R["new_records"]], .6, 1.5, 12.1, 12, [.8, 7.3, 2, 2])
tb(s, .6, 4.9, 12, 1.8, ["App: 6 páginas (problema, EDA, base vs optimizada, parámetros, evaluación final, formulario).", "Carga el Pipeline con joblib; no busca hiperparámetros por consulta."], 16)

s = slide("14. Conclusiones y limitaciones", "Cierre y preguntas. Limitaciones: datos sintéticos, costos supuestos, pocas fallas.")
bullets(s, [f"{W} fue el mejor por CV (PR-AUC {f3(TS[W]['pr_auc_cv'])}); en Test: PR-AUC {f3(TE['pr_auc'])}, recall {f3(TE['recall'])}.",
            "CV dentro de Train + Test intacto = estimación honesta.", "Umbral por costo alinea el modelo con el negocio.",
            "La métrica elegida condiciona al ganador: se justificó antes del ajuste y se reportó la sensibilidad.",
            "Limitaciones: datos sintéticos, pocas fallas en Test (IC amplios), costos 10:1 supuestos.",
            "En entorno real: validación cronológica, calibración, variables derivadas, monitoreo de deriva."], size=17)
os.makedirs("presentacion", exist_ok=True)
prs.save("presentacion/Presentacion_Grupo09.pptx"); print("Presentación generada:", len(prs.slides), "diapositivas")
