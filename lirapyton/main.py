from fastapi import FastAPI, Response
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')

# Configuración MongoDB
MONGO_URI = "mongodb+srv://arialvarado22s:0000@cluster0.abppkfd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "lira"
COLLECTION_NAME = "datos"

app = FastAPI()

# Conexión a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def calcular_edad(fecha_nacimiento):
    """Calcula la edad a partir de la fecha de nacimiento"""
    hoy = datetime.utcnow()
    return hoy.year - fecha_nacimiento.year - ((hoy.month, hoy.day) < (fecha_nacimiento.month, fecha_nacimiento.day))

def obtener_datos_usuarios():
    """Función auxiliar para obtener y procesar datos de usuarios"""
    pipeline = [
        {"$project": {
            "fechaNacimiento": 1,
            "puntos": "$totalPoints",
            "genero": 1
        }}
    ]
    
    resultados_crudos = list(collection.aggregate(pipeline))
    
    if not resultados_crudos:
        return None
    
    resultados = []
    for usuario in resultados_crudos:
        if 'fechaNacimiento' in usuario:
            try:
                if isinstance(usuario['fechaNacimiento'], str):
                    fecha_nac = datetime.strptime(usuario['fechaNacimiento'], '%Y-%m-%d')
                else:
                    fecha_nac = usuario['fechaNacimiento']
                
                edad = calcular_edad(fecha_nac)
                if 6 <= edad <= 12:  # Solo niños entre 6-12 años
                    resultados.append({
                        'edad': edad,
                        'puntos': usuario.get('puntos', 0),
                        'genero': usuario.get('genero', 'No especificado')
                    })
            except Exception as e:
                print(f"Error procesando usuario: {e}")
                continue
    
    if not resultados:
        return None
        
    return pd.DataFrame(resultados)

@app.get("/")
def read_root():
    return {"Proyecto": "LIRA - Segmentación de Usuarios con 3 Algoritmos Diferentes"}

@app.get("/usuarios/segmentacion-kmeans")
def segmentacion_kmeans():
    """
    Endpoint 1: Segmentación usando K-Means Clustering
    Agrupa usuarios por edad y puntos usando K-Means
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Preparar datos para K-Means
    X = df[['edad', 'puntos']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-Means con 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Crear visualización con Plotly
    fig = px.scatter(
        df,
        x='edad',
        y='puntos',
        color='cluster',
        title="Segmentación de Usuarios - K-Means Clustering",
        labels={
            "edad": "Edad",
            "puntos": "Puntos Acumulados",
            "cluster": "Cluster"
        },
        hover_data=['genero'],
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Añadir centroides
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids):
        fig.add_scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            mode='markers',
            marker=dict(size=15, symbol='x', color='black', line=dict(width=2)),
            name=f'Centroide {i}',
            showlegend=True
        )
    
    fig.update_layout(
        template='plotly_white',
        hovermode="closest",
        xaxis_title="Edad",
        yaxis_title="Puntos Acumulados",
        legend_title="Grupos"
    )
    
    # Estadísticas por cluster
    stats = df.groupby('cluster').agg({
        'edad': ['count', 'mean'],
        'puntos': 'mean'
    }).round(2)
    
    stats.columns = ['Cantidad', 'Edad_Promedio', 'Puntos_Promedio']
    stats = stats.reset_index()
    
    # Crear tabla HTML
    table_html = stats.to_html(index=False, classes='table table-striped')
    
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <html>
        <head>
            <title>Segmentación K-Means</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Segmentación usando K-Means Clustering</h1>
            <p><strong>Algoritmo:</strong> K-Means agrupa usuarios en 4 clusters basado en edad y puntos acumulados.</p>
            {fig_html}
            <h2>Estadísticas por Cluster</h2>
            {table_html}
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

@app.get("/usuarios/segmentacion-svm")
def segmentacion_svm():
    """
    Endpoint 2: Clasificación usando Support Vector Machine (SVM)
    Clasifica usuarios en categorías de rendimiento basado en puntos
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Crear etiquetas basadas en cuartiles de puntos (rendimiento)
    df['categoria'] = pd.qcut(df['puntos'], q=4, labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
    
    # Preparar datos para SVM
    X = df[['edad', 'puntos']]
    y = df['categoria']
    
    # Estandarizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Entrenar SVM
    svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Predecir en todo el dataset
    df['prediccion'] = svm.predict(X_scaled)
    accuracy = svm.score(X_test, y_test)
    
    # Crear visualización
    fig = px.scatter(
        df,
        x='edad',
        y='puntos',
        color='prediccion',
        title=f"Clasificación SVM - Rendimiento de Usuarios (Precisión: {accuracy:.2%})",
        labels={
            "edad": "Edad",
            "puntos": "Puntos Acumulados",
            "prediccion": "Categoría de Rendimiento"
        },
        hover_data=['genero', 'categoria'],
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode="closest",
        xaxis_title="Edad",
        yaxis_title="Puntos Acumulados",
        legend_title="Categorías"
    )
    
    # Estadísticas por categoría
    stats = df.groupby('prediccion').agg({
        'edad': ['count', 'mean'],
        'puntos': 'mean'
    }).round(2)
    
    stats.columns = ['Cantidad', 'Edad_Promedio', 'Puntos_Promedio']
    stats = stats.reset_index()
    
    table_html = stats.to_html(index=False, classes='table table-striped')
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <html>
        <head>
            <title>Clasificación SVM</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Clasificación usando Support Vector Machine (SVM)</h1>
            <p><strong>Algoritmo:</strong> SVM clasifica usuarios en categorías de rendimiento basado en sus puntos acumulados.</p>
            <p><strong>Precisión del modelo:</strong> {accuracy:.2%}</p>
            {fig_html}
            <h2>Estadísticas por Categoría</h2>
            {table_html}
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

@app.get("/usuarios/segmentacion-dbscan")
def segmentacion_dbscan():
    """
    Endpoint 3: Clustering usando DBSCAN
    Identifica grupos densos y detecta usuarios atípicos (outliers)
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Preparar datos para DBSCAN
    X = df[['edad', 'puntos']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    df['cluster'] = dbscan.fit_predict(X_scaled)
    
    # Separar outliers (cluster = -1)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    n_outliers = list(df['cluster']).count(-1)
    
    # Crear visualización
    # Mapear -1 a "Outlier" para mejor visualización
    df['cluster_label'] = df['cluster'].astype(str)
    df.loc[df['cluster'] == -1, 'cluster_label'] = 'Outlier'
    
    fig = px.scatter(
        df,
        x='edad',
        y='puntos',
        color='cluster_label',
        title=f"Clustering DBSCAN - {n_clusters} Clusters + {n_outliers} Outliers",
        labels={
            "edad": "Edad",
            "puntos": "Puntos Acumulados",
            "cluster_label": "Grupo"
        },
        hover_data=['genero'],
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    
    fig.update_layout(
        template='plotly_white',
        hovermode="closest",
        xaxis_title="Edad",
        yaxis_title="Puntos Acumulados",
        legend_title="Grupos DBSCAN"
    )
    
    # Estadísticas por cluster (excluyendo outliers para algunas métricas)
    stats_all = df.groupby('cluster_label').agg({
        'edad': ['count', 'mean'],
        'puntos': 'mean'
    }).round(2)
    
    stats_all.columns = ['Cantidad', 'Edad_Promedio', 'Puntos_Promedio']
    stats_all = stats_all.reset_index()
    
    # Crear gráfico de distribución con matplotlib/seaborn
    plt.figure(figsize=(12, 5))
    
    # Distribución de clusters
    plt.subplot(1, 2, 1)
    cluster_counts = df['cluster_label'].value_counts()
    plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title('Distribución de Usuarios por Grupo')
    
    # Box plot de puntos por cluster
    plt.subplot(1, 2, 2)
    df_no_outliers = df[df['cluster'] != -1]  # Excluir outliers para mejor visualización
    if not df_no_outliers.empty:
        clusters_ordered = sorted(df_no_outliers['cluster'].unique())
        data_to_plot = [df_no_outliers[df_no_outliers['cluster'] == c]['puntos'].values 
                       for c in clusters_ordered]
        plt.boxplot(data_to_plot, labels=[f'Cluster {c}' for c in clusters_ordered])
    plt.title('Distribución de Puntos por Cluster')
    plt.ylabel('Puntos')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Guardar gráfico matplotlib
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    table_html = stats_all.to_html(index=False, classes='table table-striped')
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <html>
        <head>
            <title>Clustering DBSCAN</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
                .table th {{ background-color: #f2f2f2; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Clustering usando DBSCAN</h1>
            <p><strong>Algoritmo:</strong> DBSCAN identifica grupos densos automáticamente y detecta usuarios atípicos (outliers).</p>
            <p><strong>Resultados:</strong> {n_clusters} clusters encontrados + {n_outliers} outliers detectados</p>
            
            {fig_html}
            
            <h2>Análisis Adicional</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{img_str}" alt="Análisis DBSCAN" style="max-width: 100%; height: auto;">
            </div>
            
            <h2>Estadísticas por Grupo</h2>
            {table_html}
            
            <h3>Interpretación:</h3>
            <ul>
                <li><strong>Clusters numerados:</strong> Grupos densos de usuarios con características similares</li>
                <li><strong>Outliers:</strong> Usuarios con patrones únicos que no se ajustan a ningún grupo</li>
                <li><strong>Ventaja de DBSCAN:</strong> No requiere especificar el número de clusters previamente</li>
            </ul>
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    