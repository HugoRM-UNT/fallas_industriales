"""Ejecuta todo: entrenamiento -> informe PDF -> presentación.  Uso: python run_all.py [--n-iter 20]"""
import subprocess, sys
extra = sys.argv[1:]
for cmd in (["train.py"] + extra, ["make_report.py"], ["make_slides.py"]):
    print(">>", *cmd); subprocess.run([sys.executable, *cmd], check=True)
print("Listo. Ahora: streamlit run app.py")
