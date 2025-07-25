from fastapi import FastAPI, Response
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256, Category20
from bokeh.transform import transform
from bokeh.layouts import column, row
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

# Configurar estilo de seaborn
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.facecolor'] = 'white'

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
    return {"Proyecto": "LIRA - Segmentación de Usuarios con Visualizaciones Avanzadas"}

@app.get("/usuarios/segmentacion-plotly")
def segmentacion_plotly():
    """
    Endpoint 1: Segmentación usando Plotly con diseño avanzado
    Visualizaciones interactivas y profesionales
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
    
    # Crear subplot con múltiples visualizaciones
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Segmentación por Clusters', 'Distribución por Edad', 
                       'Distribución por Género', 'Análisis de Puntos'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # Colores modernos y atractivos
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 1. Scatter plot principal
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['edad'],
                y=cluster_data['puntos'],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(
                    color=colors[i],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Edad:</b> %{x}<br><b>Puntos:</b> %{y}<br><extra></extra>'
            ),
            row=1, col=1
        )
    
    # Añadir centroides
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids):
        fig.add_trace(
            go.Scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                mode='markers',
                name=f'Centro {i}',
                marker=dict(
                    color='black',
                    size=15,
                    symbol='x',
                    line=dict(width=3, color=colors[i])
                ),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Distribución por edad
    edad_counts = df.groupby(['edad', 'cluster']).size().reset_index(name='count')
    for i in range(4):
        edad_cluster = edad_counts[edad_counts['cluster'] == i]
        fig.add_trace(
            go.Bar(
                x=edad_cluster['edad'],
                y=edad_cluster['count'],
                name=f'Cluster {i}',
                marker_color=colors[i],
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Distribución por género
    genero_counts = df.groupby(['genero', 'cluster']).size().reset_index(name='count')
    for i in range(4):
        genero_cluster = genero_counts[genero_counts['cluster'] == i]
        fig.add_trace(
            go.Bar(
                x=genero_cluster['genero'],
                y=genero_cluster['count'],
                name=f'Cluster {i}',
                marker_color=colors[i],
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Box plot de puntos por cluster
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        fig.add_trace(
            go.Box(
                y=cluster_data['puntos'],
                name=f'Cluster {i}',
                marker_color=colors[i],
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Personalizar layout
    fig.update_layout(
        title={
            'text': "🎯 Análisis Avanzado de Segmentación - LIRA",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2C3E50'}
        },
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial, sans-serif", size=12),
        paper_bgcolor='rgba(248,249,250,1)',
        plot_bgcolor='white'
    )
    
    # Estadísticas mejoradas
    stats = df.groupby('cluster').agg({
        'edad': ['count', 'mean', 'std'],
        'puntos': ['mean', 'std', 'median']
    }).round(2)
    
    stats.columns = ['Usuarios', 'Edad_Media', 'Edad_Std', 'Puntos_Media', 'Puntos_Std', 'Puntos_Mediana']
    stats = stats.reset_index()
    
    table_html = stats.to_html(index=False, classes='table table-hover table-striped')
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Segmentación Avanzada - Plotly</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin: 20px auto;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    color: #2C3E50;
                    margin-bottom: 30px;
                }}
                .stats-card {{
                    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .table {{ border-radius: 10px; overflow: hidden; }}
                .badge {{ font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Segmentación Inteligente con Plotly</h1>
                    <div class="row">
                        <div class="col-md-3"><span class="badge bg-primary">K-Means Clustering</span></div>
                        <div class="col-md-3"><span class="badge bg-success">4 Clusters</span></div>
                        <div class="col-md-3"><span class="badge bg-warning">Usuarios: {len(df)}</span></div>
                        <div class="col-md-3"><span class="badge bg-info">Edades: 6-12 años</span></div>
                    </div>
                </div>
                
                {fig_html}
                
                <div class="stats-card">
                    <h3>📊 Estadísticas Detalladas por Cluster</h3>
                    {table_html}
                </div>
                
                <div class="alert alert-info">
                    <h5>💡 Insights del Análisis:</h5>
                    <ul>
                        <li><strong>Visualización interactiva:</strong> Hover sobre los puntos para ver detalles</li>
                        <li><strong>Centroides marcados:</strong> Los puntos negros muestran el centro de cada grupo</li>
                        <li><strong>Múltiples perspectivas:</strong> Análisis por edad, género y rendimiento</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

@app.get("/usuarios/segmentacion-seaborn")
def segmentacion_seaborn():
    """
    Endpoint 2: Visualizaciones estadísticas avanzadas con Seaborn
    Análisis estadístico profundo y elegante
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Aplicar clustering
    X = df[['edad', 'puntos']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📈 Análisis Estadístico Avanzado con Seaborn', fontsize=20, fontweight='bold')
    
    # Configurar paleta de colores moderna
    colors = sns.color_palette("husl", 4)
    
    # 1. Scatter plot con regresión
    sns.scatterplot(data=df, x='edad', y='puntos', hue='cluster', 
                   palette=colors, s=100, alpha=0.8, ax=axes[0,0])
    sns.regplot(data=df, x='edad', y='puntos', scatter=False, 
               color='red', ax=axes[0,0], line_kws={'linewidth': 2})
    axes[0,0].set_title('🎯 Clusters con Tendencia General', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Violin plot por cluster
    sns.violinplot(data=df, x='cluster', y='puntos', palette=colors, ax=axes[0,1])
    axes[0,1].set_title('🎻 Distribución de Puntos por Cluster', fontweight='bold')
    axes[0,1].set_xlabel('Cluster')
    
    # 3. Heatmap de correlación
    corr_data = df[['edad', 'puntos', 'cluster']].corr()
    sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
               square=True, ax=axes[0,2], cbar_kws={'shrink': 0.8})
    axes[0,2].set_title('🔥 Matriz de Correlación', fontweight='bold')
    
    # 4. Box plot por género y cluster
    if len(df['genero'].unique()) > 1:
        sns.boxplot(data=df, x='genero', y='puntos', hue='cluster', 
                   palette=colors, ax=axes[1,0])
        axes[1,0].set_title('📊 Puntos por Género y Cluster', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        sns.histplot(data=df, x='puntos', hue='cluster', multiple="stack", 
                    palette=colors, ax=axes[1,0])
        axes[1,0].set_title('📊 Distribución de Puntos', fontweight='bold')
    
    # 5. Strip plot con swarm
    sns.swarmplot(data=df, x='cluster', y='edad', palette=colors, 
                 size=8, alpha=0.8, ax=axes[1,1])
    axes[1,1].set_title('🐝 Distribución de Edades por Cluster', fontweight='bold')
    
    # 6. Pair plot style dentro del subplot
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        axes[1,2].scatter(cluster_data['edad'], cluster_data['puntos'], 
                         color=colors[i], label=f'Cluster {cluster}', 
                         s=60, alpha=0.7, edgecolors='white', linewidth=1)
    
    axes[1,2].set_xlabel('Edad')
    axes[1,2].set_ylabel('Puntos')
    axes[1,2].set_title('🎨 Vista Detallada por Clusters', fontweight='bold')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar imagen
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Análisis estadístico adicional
    stats_summary = df.groupby('cluster').agg({
        'edad': ['count', 'mean', 'std', 'min', 'max'],
        'puntos': ['mean', 'std', 'min', 'max', 'median']
    }).round(2)
    
    # Crear segunda visualización: análisis de distribuciones
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribución de puntos por cluster
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]['puntos']
        sns.histplot(cluster_data, alpha=0.7, label=f'Cluster {cluster}', 
                    color=colors[i], ax=axes2[0])
    
    axes2[0].set_title('📊 Distribución de Puntos Superpuesta', fontweight='bold', fontsize=14)
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Análisis de densidad
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]['puntos']
        sns.kdeplot(cluster_data, label=f'Cluster {cluster}', 
                   color=colors[i], linewidth=3, ax=axes2[1])
    
    axes2[1].set_title('🌊 Curvas de Densidad por Cluster', fontweight='bold', fontsize=14)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    buffer2.seek(0)
    plt.close()
    
    img_str2 = base64.b64encode(buffer2.read()).decode('utf-8')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Análisis Seaborn - LIRA</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ 
                    font-family: 'Segoe UI', system-ui, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    margin: 20px auto;
                    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 15px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .insights {{
                    background: linear-gradient(45deg, #00d2d3, #54a0ff);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Análisis Estadístico Profesional</h1>
                    <h3>Powered by Seaborn & Statistical Analysis</h3>
                    <p>Visualizaciones elegantes para insights profundos</p>
                </div>
                
                <div class="chart-container">
                    <h3>🎨 Suite Completa de Visualizaciones</h3>
                    <img src="data:image/png;base64,{img_str}" alt="Análisis Seaborn" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                </div>
                
                <div class="chart-container">
                    <h3>📈 Análisis de Distribuciones Avanzado</h3>
                    <img src="data:image/png;base64,{img_str2}" alt="Distribuciones" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>👥 Total Usuarios</h4>
                        <h2>{len(df)}</h2>
                    </div>
                    <div class="stat-card">
                        <h4>🎯 Clusters</h4>
                        <h2>4</h2>
                    </div>
                    <div class="stat-card">
                        <h4>📊 Variables</h4>
                        <h2>Edad + Puntos</h2>
                    </div>
                    <div class="stat-card">
                        <h4>🎨 Visualizaciones</h4>
                        <h2>8 Gráficos</h2>
                    </div>
                </div>
                
                <div class="insights">
                    <h3>🔍 Características del Análisis Seaborn:</h3>
                    <div class="row">
                        <div class="col-md-6">
                            <ul>
                                <li><strong>Violin Plots:</strong> Muestran distribución completa</li>
                                <li><strong>Swarm Plots:</strong> Cada punto es visible sin superposición</li>
                                <li><strong>Regresión:</strong> Tendencias automáticas</li>
                                <li><strong>Correlaciones:</strong> Heatmap intuitivo</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <ul>
                                <li><strong>Box Plots:</strong> Quartiles y outliers claros</li>
                                <li><strong>KDE:</strong> Curvas de densidad suaves</li>
                                <li><strong>Histogramas:</strong> Distribuciones superpuestas</li>
                                <li><strong>Paletas:</strong> Colores estadísticamente apropiados</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

@app.get("/usuarios/segmentacion-bokeh")
def segmentacion_bokeh():
    """
    Endpoint 3: Visualizaciones interactivas avanzadas con Bokeh
    Dashboards interactivos y modernos
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Aplicar clustering
    X = df[['edad', 'puntos']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Colores para clusters
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    df['color'] = df['cluster'].map({i: colors[i] for i in range(4)})
    
    # 1. Scatter plot principal con interactividad avanzada
    p1 = figure(
        title="🎯 Segmentación Interactiva de Usuarios",
        x_axis_label="Edad",
        y_axis_label="Puntos Acumulados",
        width=700,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        toolbar_location="above"
    )
    
    # Añadir puntos por cluster
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        source = dict(
            x=cluster_data['edad'].tolist(),
            y=cluster_data['puntos'].tolist(),
            genero=cluster_data['genero'].tolist(),
            cluster=[f'Cluster {i}'] * len(cluster_data)
        )
        
        scatter = p1.circle(
            'x', 'y', 
            size=12, 
            color=colors[i], 
            alpha=0.8,
            legend_label=f'Cluster {i}',
            source=source,
            line_color='white',
            line_width=2
        )
    
    # Configurar hover tool personalizado
    hover = p1.select_one(HoverTool)
    hover.tooltips = [
        ("Cluster", "@cluster"),
        ("Edad", "@x años"),
        ("Puntos", "@y"),
        ("Género", "@genero")
    ]
    
    # Añadir centroides
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids):
        p1.x(centroid[0], centroid[1], size=20, color='black', 
             line_color=colors[i], line_width=3, alpha=0.9)
    
    # Personalizar apariencia
    p1.title.text_font_size = "16pt"
    p1.title.text_color = "#2C3E50"
    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"
    p1.background_fill_color = "#F8F9FA"
    p1.border_fill_color = "white"
    
    # 2. Histograma interactivo de edades
    p2 = figure(
        title="📊 Distribución de Edades por Cluster",
        x_axis_label="Edad",
        y_axis_label="Cantidad de Usuarios",
        width=700,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Crear histograma apilado
    edad_counts = df.groupby(['edad', 'cluster']).size().reset_index(name='count')
    
    # Preparar datos para histograma apilado
    edades = sorted(df['edad'].unique())
    bottom = np.zeros(len(edades))
    
    for i in range(4):
        cluster_counts = []
        for edad in edades:
            count = edad_counts[(edad_counts['edad'] == edad) & 
                              (edad_counts['cluster'] == i)]['count'].sum()
            cluster_counts.append(count)
        
        p2.vbar(x=edades, top=cluster_counts, bottom=bottom, width=0.8,
               color=colors[i], legend_label=f'Cluster {i}', alpha=0.8)
        bottom += cluster_counts
    
    p2.legend.location = "top_right"
    p2.background_fill_color = "#F8F9FA"
    
    # 3. Gráfico de puntos vs edad con líneas de tendencia
    p3 = figure(
        title="📈 Análisis de Tendencias por Cluster",
        x_axis_label="Edad",
        y_axis_label="Puntos Promedio",
        width=700,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Calcular promedios por edad y cluster
    avg_points = df.groupby(['edad', 'cluster'])['puntos'].mean().reset_index()
    
    for i in range(4):
        cluster_avg = avg_points[avg_points['cluster'] == i]
        if not cluster_avg.empty:
            # Líneas de tendencia
            p3.line(cluster_avg['edad'], cluster_avg['puntos'], 
                   line_width=3, color=colors[i], alpha=0.8,
                   legend_label=f'Cluster {i}')
            # Puntos en las líneas
            p3.circle(cluster_avg['edad'], cluster_avg['puntos'], 
                     size=10, color=colors[i], alpha=0.9,
                     line_color='white', line_width=2)
    
    p3.legend.location = "top_left"
    p3.background_fill_color = "#F8F9FA"
    
    # 4. Box plot usando Bokeh (simulado con percentiles)
    p4 = figure(
        title="📦 Análisis de Dispersión por Cluster",
        x_axis_label="Cluster",
        y_axis_label="Puntos",
        width=700,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Calcular estadísticas para box plot
    for i in range(4):
        cluster_data = df[df['cluster'] == i]['puntos']
        if not cluster_data.empty:
            q1 = cluster_data.quantile(0.25)
            q2 = cluster_data.quantile(0.5)  # mediana
            q3 = cluster_data.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            
            # Caja principal
            p4.quad(left=i-0.3, right=i+0.3, bottom=q1, top=q3,
                   color=colors[i], alpha=0.7, line_color='black')
            
            # Línea de mediana
            p4.line([i-0.3, i+0.3], [q2, q2], line_width=3, color='black')
            
            # Bigotes
            p4.line([i, i], [q3, min(upper, cluster_data.max())], 
                   line_width=2, color='black')
            p4.line([i, i], [q1, max(lower, cluster_data.min())], 
                   line_width=2, color='black')
            
            # Outliers
            outliers = cluster_data[(cluster_data > upper) | (cluster_data < lower)]
            if not outliers.empty:
                p4.circle([i] * len(outliers), outliers, 
                         size=8, color=colors[i], alpha=0.6)
    
    p4.xaxis.ticker = [0, 1, 2, 3]
    p4.xaxis.major_label_overrides = {0: "Cluster 0", 1: "Cluster 1", 
                                     2: "Cluster 2", 3: "Cluster 3"}
    p4.background_fill_color = "#F8F9FA"
    
    # Crear layout con múltiples gráficos
    layout = column(
        row(p1, p2),
        row(p3, p4)
    )
    
    # Generar componentes HTML
    script, div = components(layout)
    
    # Estadísticas para la tabla
    stats = df.groupby('cluster').agg({
        'edad': ['count', 'mean', 'std'],
        'puntos': ['mean', 'std', 'median']
    }).round(2)
    
    stats.columns = ['Usuarios', 'Edad_Media', 'Edad_Std', 'Puntos_Media', 'Puntos_Std', 'Puntos_Mediana']
    stats = stats.reset_index()
    
    table_html = stats.to_html(index=False, classes='table table-hover table-striped')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Dashboard Interactivo - Bokeh</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.3.min.js"></script>
            <style>
                body {{ 
                    font-family: 'Segoe UI', system-ui, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    margin: 0 auto;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    max-width: 1600px;
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    padding: 40px;
                    border-radius: 15px;
                    margin-bottom: 40px;
                    position: relative;
                    overflow: hidden;
                }}
                .header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: repeating-linear-gradient(
                        45deg,
                        transparent,
                        transparent 10px,
                        rgba(255,255,255,0.1) 10px,
                        rgba(255,255,255,0.1) 20px
                    );
                    animation: slide 20s linear infinite;
                }}
                @keyframes slide {{
                    0% {{ transform: translateX(-50px) translateY(-50px); }}
                    100% {{ transform: translateX(50px) translateY(50px); }}
                }}
                .dashboard-grid {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 15px;
                    margin: 30px 0;
                }}
                .stats-container {{
                    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    margin: 30px 0;
                }}
                .feature-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .feature-card {{
                    background: linear-gradient(45deg, #4ecdc4, #44a08d);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    transform: translateY(0);
                    transition: transform 0.3s ease;
                }}
                .feature-card:hover {{
                    transform: translateY(-5px);
                }}
                .bk-root {{
                    background: #f8f9fa !important;
                    border-radius: 15px !important;
                    padding: 20px !important;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Dashboard Interactivo Avanzado</h1>
                    <h3>Powered by Bokeh - Visualizaciones de Próxima Generación</h3>
                    <p style="position: relative; z-index: 1;">Experiencia inmersiva con interactividad completa</p>
                </div>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <h4>🎯 Hover Interactivo</h4>
                        <p>Información detallada al pasar el mouse</p>
                    </div>
                    <div class="feature-card">
                        <h4>🔍 Zoom Dinámico</h4>
                        <p>Explora los datos con zoom y pan</p>
                    </div>
                    <div class="feature-card">
                        <h4>👁️ Filtros Visuales</h4>
                        <p>Click en la leyenda para ocultar/mostrar</p>
                    </div>
                    <div class="feature-card">
                        <h4>📊 Múltiples Vistas</h4>
                        <p>4 visualizaciones complementarias</p>
                    </div>
                </div>
                
                <div class="dashboard-grid">
                    <h2 style="text-align: center; color: #2C3E50; margin-bottom: 30px;">
                        📈 Suite Completa de Visualizaciones Interactivas
                    </h2>
                    {div}
                </div>
                
                <div class="stats-container">
                    <h3>📊 Estadísticas Detalladas del Clustering</h3>
                    <div class="row">
                        <div class="col-md-8">
                            {table_html}
                        </div>
                        <div class="col-md-4">
                            <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;">
                                <h5>🎯 Características Bokeh:</h5>
                                <ul>
                                    <li><strong>Interactividad Real:</strong> Herramientas de navegación</li>
                                    <li><strong>Hover Personalizado:</strong> Información contextual</li>
                                    <li><strong>Leyendas Activas:</strong> Click para filtrar</li>
                                    <li><strong>Zoom Inteligente:</strong> Enfoque en áreas específicas</li>
                                    <li><strong>Export Ready:</strong> Guarda como imagen</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="alert alert-info" style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; border-radius: 15px;">
                    <h4>💡 Cómo Usar el Dashboard:</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <ul>
                                <li><strong>Scatter Plot:</strong> Hover para detalles, zoom para explorar</li>
                                <li><strong>Histograma:</strong> Distribución apilada por cluster</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <ul>
                                <li><strong>Tendencias:</strong> Líneas conectan promedios por edad</li>
                                <li><strong>Box Plot:</strong> Dispersión y outliers por cluster</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            {script}
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

# Endpoint adicional con comparación de las 3 librerías
@app.get("/usuarios/comparacion-librerias")
def comparacion_librerias():
    """
    Endpoint especial: Comparación visual de las 3 mejores librerías
    Muestra las fortalezas de cada una lado a lado
    """
    df = obtener_datos_usuarios()
    
    if df is None or df.empty:
        return Response(content="No hay datos disponibles para análisis.", media_type="text/html")
    
    # Aplicar clustering una vez para todas las visualizaciones
    X = df[['edad', 'puntos']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 1. PLOTLY - Gráfico 3D interactivo
    fig_plotly = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        fig_plotly.add_trace(go.Scatter3d(
            x=cluster_data['edad'],
            y=cluster_data['puntos'],
            z=cluster_data['cluster'],
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(
                color=colors[i],
                size=8,
                opacity=0.8
            ),
            hovertemplate='<b>Edad:</b> %{x}<br><b>Puntos:</b> %{y}<br><b>Cluster:</b> %{z}<extra></extra>'
        ))
    
    fig_plotly.update_layout(
        title='🎯 PLOTLY: Visualización 3D Interactiva',
        scene=dict(
            xaxis_title='Edad',
            yaxis_title='Puntos',
            zaxis_title='Cluster',
            bgcolor='rgba(240,240,240,0.8)'
        ),
        height=600,
        font=dict(size=12)
    )
    
    plotly_html = fig_plotly.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 2. SEABORN - Análisis estadístico elegante
    plt.style.use('seaborn-v0_8')
    fig_seaborn, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pair plot style
    sns.scatterplot(data=df, x='edad', y='puntos', hue='cluster', 
                   palette='husl', s=100, alpha=0.8, ax=axes[0])
    axes[0].set_title('🎨 SEABORN: Scatter con Regresión', fontweight='bold', fontsize=14)
    
    # Ridge plot simulation
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]['puntos']
        sns.kdeplot(cluster_data, label=f'Cluster {cluster}', 
                   linewidth=3, ax=axes[1])
    
    axes[1].set_title('📊 SEABORN: Densidades Superpuestas', fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer_seaborn = BytesIO()
    plt.savefig(buffer_seaborn, format='png', dpi=300, bbox_inches='tight',
               facecolor='white')
    buffer_seaborn.seek(0)
    plt.close()
    
    seaborn_img = base64.b64encode(buffer_seaborn.read()).decode('utf-8')
    
    # 3. BOKEH - Dashboard minimalista
    p_bokeh = figure(
        title="⚡ BOKEH: Dashboard Interactivo Minimalista",
        x_axis_label="Edad",
        y_axis_label="Puntos",
        width=700,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,hover",
        toolbar_location="above"
    )
    
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        p_bokeh.circle(
            cluster_data['edad'], cluster_data['puntos'],
            size=12, color=colors[i], alpha=0.7,
            legend_label=f'Cluster {i}',
            line_color='white', line_width=2
        )
    
    p_bokeh.legend.click_policy = "hide"
    p_bokeh.background_fill_color = "#F8F9FA"
    
    bokeh_script, bokeh_div = components(p_bokeh)
    
    # Crear tabla comparativa
    comparison_data = {
        'Característica': [
            'Interactividad',
            'Facilidad de Uso',
            'Personalización',
            'Análisis Estadístico',
            'Exportación',
            'Rendimiento',
            'Comunidad',
            'Documentación'
        ],
        'Plotly': [
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐'
        ],
        'Seaborn': [
            '⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐'
        ],
        'Bokeh': [
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_table = comparison_df.to_html(index=False, classes='table table-striped table-hover')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Comparación de Librerías - LIRA</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.3.min.js"></script>
            <style>
                body {{ 
                    font-family: 'Segoe UI', system-ui, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    border-radius: 25px;
                    padding: 50px;
                    margin: 0 auto;
                    box-shadow: 0 25px 50px rgba(0,0,0,0.1);
                    max-width: 1800px;
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 50px;
                    border-radius: 20px;
                    margin-bottom: 50px;
                    position: relative;
                    overflow: hidden;
                }}
                .header::before {{
                    content: '📊📈📉📋';
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    font-size: 30px;
                    opacity: 0.3;
                    animation: float 3s ease-in-out infinite;
                }}
                @keyframes float {{
                    0%, 100% {{ transform: translateY(0px); }}
                    50% {{ transform: translateY(-10px); }}
                }}
                .library-section {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 20px;
                    margin: 30px 0;
                    border-left: 5px solid;
                }}
                .plotly-section {{ border-left-color: #FF6B6B; }}
                .seaborn-section {{ border-left-color: #4ECDC4; }}
                .bokeh-section {{ border-left-color: #45B7D1; }}
                .comparison-table {{
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    padding: 40px;
                    border-radius: 20px;
                    margin: 40px 0;
                }}
                .table {{
                    background: white;
                    border-radius: 15px;
                    overflow: hidden;
                }}
                .table th {{
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    color: white;
                    border: none;
                    padding: 15px;
                }}
                .table td {{
                    padding: 12px 15px;
                    border: none;
                    border-bottom: 1px solid #eee;
                }}
                .pros-cons {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 30px;
                    margin: 30px 0;
                }}
                .pros-cons-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .pros {{ border-left: 4px solid #27ae60; }}
                .cons {{ border-left: 4px solid #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 Comparación de las 3 Mejores Librerías de Visualización</h1>
                    <h3>Plotly vs Seaborn vs Bokeh</h3>
                    <p>Análisis exhaustivo para elegir la herramienta perfecta</p>
                </div>
                
                <!-- PLOTLY SECTION -->
                <div class="library-section plotly-section">
                    <h2>🎯 PLOTLY - El Rey de la Interactividad</h2>
                    <div class="row">
                        <div class="col-md-8">
                            {plotly_html}
                        </div>
                        <div class="col-md-4">
                            <div class="pros-cons">
                                <div class="pros-cons-card pros">
                                    <h5>✅ Fortalezas:</h5>
                                    <ul>
                                        <li>Interactividad nativa</li>
                                        <li>Gráficos 3D increíbles</li>
                                        <li>Export a múltiples formatos</li>
                                        <li>Dashboards web listos</li>
                                        <li>Animaciones fluidas</li>
                                    </ul>
                                </div>
                                <div class="pros-cons-card cons">
                                    <h5>❌ Limitaciones:</h5>
                                    <ul>
                                        <li>Curva de aprendizaje</li>
                                        <li>Tamaño de archivos</li>
                                        <li>Dependencias JavaScript</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- SEABORN SECTION -->
                <div class="library-section seaborn-section">
                    <h2>🎨 SEABORN - Elegancia Estadística</h2>
                    <div class="row">
                        <div class="col-md-8">
                            <img src="data:image/png;base64,{seaborn_img}" 
                                 alt="Seaborn Analysis" 
                                 style="max-width: 100%; height: auto; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                        </div>
                        <div class="col-md-4">
                            <div class="pros-cons">
                                <div class="pros-cons-card pros">
                                    <h5>✅ Fortalezas:</h5>
                                    <ul>
                                        <li>Sintaxis súper simple</li>
                                        <li>Integración con Pandas</li>
                                        <li>Gráficos estadísticos nativos</li>
                                        <li>Temas hermosos por defecto</li>
                                        <li>Ideal para análisis exploratorio</li>
                                    </ul>
                                </div>
                                <div class="pros-cons-card cons">
                                    <h5>❌ Limitaciones:</h5>
                                    <ul>
                                        <li>Sin interactividad</li>
                                        <li>Personalización limitada</li>
                                        <li>Depende de Matplotlib</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- BOKEH SECTION -->
                <div class="library-section bokeh-section">
                    <h2>⚡ BOKEH - Dashboards Profesionales</h2>
                    <div class="row">
                        <div class="col-md-8">
                            {bokeh_div}
                        </div>
                        <div class="col-md-4">
                            <div class="pros-cons">
                                <div class="pros-cons-card pros">
                                    <h5>✅ Fortalezas:</h5>
                                    <ul>
                                        <li>Aplicaciones web completas</li>
                                        <li>Manejo de Big Data</li>
                                        <li>Streaming en tiempo real</li>
                                        <li>Widgets interactivos</li>
                                        <li>Servidor Bokeh integrado</li>
                                    </ul>
                                </div>
                                <div class="pros-cons-card cons">
                                    <h5>❌ Limitaciones:</h5>
                                    <ul>
                                        <li>Sintaxis más compleja</li>
                                        <li>Curva de aprendizaje empinada</li>
                                        <li>Documentación dispersa</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- COMPARISON TABLE -->
                <div class="comparison-table">
                    <h2 style="text-align: center; margin-bottom: 30px;">📊 Tabla Comparativa Detallada</h2>
                    {comparison_table}
                </div>
                
                <!-- RECOMMENDATIONS -->
                <div class="alert" style="background: linear-gradient(45deg, #27ae60, #2ecc71); color: white; border: none; border-radius: 20px; padding: 30px;">
                    <h3>🎯 Recomendaciones por Caso de Uso:</h3>
                    <div class="row">
                        <div class="col-md-4">
                            <h5>📊 Para Análisis Exploratorio:</h5>
                            <p><strong>SEABORN</strong> - Rápido, elegante y estadísticamente robusto</p>
                        </div>
                        <div class="col-md-4">
                            <h5>🌐 Para Dashboards Web:</h5>
                            <p><strong>PLOTLY</strong> - Interactividad nativa y facilidad de deployment</p>
                        </div>
                        <div class="col-md-4">
                            <h5>⚡ Para Aplicaciones Complejas:</h5>
                            <p><strong>BOKEH</strong> - Máximo control y escalabilidad</p>
                        </div>
                    </div>
                </div>
            </div>
            
            {bokeh_script}
        </body>
    </html>
    """
    
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)