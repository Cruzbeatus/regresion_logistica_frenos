from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from io import StringIO

app = FastAPI()

# --- Fase 2: Cargar y arreglar los datos ---
with open("falla_frenos.csv", "r", encoding="utf-8") as f:
    raw = f.read().replace('"', '')

df = pd.read_csv(StringIO(raw))

print("Columnas detectadas:", df.columns.tolist())
print(df.head())

# --- Fase 3: Preparación de los datos ---
X = df[['kms_recorridos', 'años_uso', 'ultima_revision', 'temperatura_frenos',
        'cambios_pastillas', 'estilo_conduccion', 'carga_promedio', 'luz_alarma_freno']]
y = df['falla_frenos']

# --- Fase 4: Modelado ---
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X, y)

# --- Fase 5: Evaluación ---
predicciones = modelo.predict(X)
precision = accuracy_score(y, predicciones)
print(f"Precisión del modelo: {precision*100:.2f}%")


# --- Fase 6: Implementación API ---

@app.get("/")
def home():
    return {"mensaje": "API de predicción de fallas en frenos"}

@app.post("/predecir/")
def predecir(datos: dict):
    nuevo_vehiculo = [[
        datos["kms_recorridos"],
        datos["años_uso"],
        datos["ultima_revision"],
        datos["temperatura_frenos"],
        datos["cambios_pastillas"],
        datos["estilo_conduccion"],
        datos["carga_promedio"],
        datos["luz_alarma_freno"]
    ]]
    prediccion = modelo.predict(nuevo_vehiculo)[0]
    return {
        "prediccion": int(prediccion),
        "precision_entrenamiento": f"{precision*100:.2f}%"
    }


# --- prueba ---
#nuevo_vehiculo = [[120000, 10, 15, 80, 4, 1, 650, 1]]
#prediccion = modelo.predict(nuevo_vehiculo)[0]
#1
#if prediccion == 1:
#    print("La predicción es: El vehículo se quedará sin frenos (1)")
#else:
#    print("La predicción es: El vehículo no se quedará sin frenos (0)")
