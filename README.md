# AdventureWorks Analytics — EDA + K-Means (módulo Sales)

Análisis exploratorio del esquema `Sales` de **AdventureWorks2022** y segmentación de clientes
individuales con **K-Means sobre variables RFM** (Recency, Frequency, Monetary).

## Estructura

```
AdventureWorks-Analytics/
├── .env                      # credenciales de SQL Server (no se sube a git)
├── src/
│   ├── db.py                 # conexión (pymssql + SQLAlchemy)
│   ├── extract.py            # SQL Server -> data/raw/*.parquet
│   ├── features.py           # limpieza + variables RFM por cliente
│   ├── clustering.py         # escalado, evaluación de k, K-Means, PCA, nombres de segmentos
│   └── viz.py                # estilo y paleta de gráficos
├── notebooks/
│   ├── 01_EDA_Sales.ipynb            # análisis exploratorio + conclusiones
│   └── 02_Clustering_KMeans.ipynb    # elección de k, modelo, perfiles, recomendaciones
├── app/streamlit_app.py      # dashboard interactivo
├── data/raw | data/processed # parquet generados
└── reports/                  # CSV de clientes segmentados y perfil de segmentos
```

## Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # y pon la contraseña real

python -m src.extract         # 1. descarga los datos de SQL Server (una sola vez)
jupyter notebook notebooks    # 2. abre 01_EDA y 02_Clustering
streamlit run app/streamlit_app.py   # 3. dashboard en http://localhost:8501
```

Después de `extract`, todo funciona **sin conexión** a la base (usa los `.parquet`).

## Resultados principales (k = 4)

| Segmento | % clientes | % ventas B2C | Perfil |
|---|---|---|---|
| Campeones | 30 % | 77 % | ~2,4 pedidos, ≈ $4.000, compran bicicletas |
| Alto valor en riesgo | 20 % | 20 % | 1 bicicleta hace ~7 meses |
| Nuevos | 14 % | 3 % | compra reciente (~1 mes), accesorios |
| Bajo valor inactivos | 36 % | 1 % | 1 compra de ≈ $50 en accesorios/ropa |

Silhouette medio = 0,41. Detalle e interpretación en `notebooks/02_Clustering_KMeans.ipynb`.
