import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import matplotlib.dates as mdates
from scipy import stats
import re
import networkx as nx
import random

pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', None)

from scipy.stats import f_oneway, chi2_contingency

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, calinski_harabasz_score, davies_bouldin_score

def estudio_univariante(df: pd.DataFrame, col: str) -> None:
    '''
    Esta función imprime tres subplots (Histograma, Boxplot y KDE plot) y imprime
    el sesgo de la distribución.
        Parámetros:
        col: es un parámetro de tipo cadena, que sirve para especificar a que
             variable se le hará el estudio.
        Returns:
            Nada, imprime desde la función.
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma
    ax1.hist(df[col], color='red')
    ax1.set_xlabel(col)
    ax1.set_ylabel('Cantidad')
    ax1.set_title(f'Histograma de la variable {col}')

    # Boxplot
    ax2.boxplot(df[col])
    ax2.set_xlabel(col)
    ax2.set_title(f'Boxplot de {col}')

    #KDE plot
    sns.kdeplot(ax = ax3, data=df, x=df[col], fill= True, color='red')
    ax3.set_title(f'KDE plot de {col}')
    ax3.set_xlabel(col)
    ax3.set_ylabel('Densidad')
    plt.show()

def timeline(data, asset):
    filtro = data.loc[data['asset'] == asset].copy()
    filtro_mes_tren = filtro.groupby('Fecha')['EnergiaConsumidaTotal'].max().reset_index()


    fecha_inicio = pd.to_datetime('2024-02-15')
    fecha_fin = pd.to_datetime('2024-03-15')
    todas_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')

    filtro_mes_tren = filtro_mes_tren.set_index('Fecha').reindex(todas_fechas).reset_index()
    filtro_mes_tren = filtro_mes_tren.rename(columns={'index': 'Fecha'})


    plt.figure(figsize=(12, 6))
    sns.lineplot(data=filtro_mes_tren, x='Fecha', y='EnergiaConsumidaTotal')
    plt.title(f'Consumo de Energía Diario {asset}')
    plt.xlabel('Fecha')
    plt.ylabel('Energía Consumida Total (Máximo)')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def timeline_por_componente(data, trenes, componentes, titulo) -> None:
    for tren in trenes:
        # Filtrar el dataset por cada tren específico
        filtro_tren = data[data['asset'] == tren]

        # Crear el gráfico
        plt.figure(figsize=(15, 6))

        for componente in componentes:
            # Filtrar por el componente específico


            # Encontrar el valor máximo de
            filtro_componente = filtro_tren[['Fecha', componente]].groupby('Fecha')[componente].max().reset_index()

            # Graficar la línea temporal para el componente actual
            sns.lineplot(data=filtro_componente, x='Fecha', y=componente, label=componente)

        # Títulos y etiquetas
        plt.title(f'Consumo de Energía Diario para {tren}, por componente {titulo}')
        plt.xlabel('Fecha')
        plt.ylabel('Energía Consumida Total por Día')

        # Configuración del eje x (fechas)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

        # Rotar las etiquetas de fecha en el eje x
        plt.xticks(rotation=45, ha='right')

        plt.legend(title='Componente de Tracción')
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()

def dibujar_arbol_recorrido(df, fecha, tren):
    plt.figure(figsize=(6, 4))
    df_tren_fecha = df[(df['Fecha'] == fecha) & (df['asset'] == tren)]

    if df_tren_fecha.empty:
        print(f"No hay registros disponibles para el Tren {tren} en la fecha {fecha}.")
        return

    
    df_tren_fecha = df_tren_fecha.sort_values(by='FechaHora')
    G = nx.DiGraph()

    
    for i, row in df_tren_fecha.iterrows():
        atc_cs = row['ATC_CS']  
        atc_ds = row['ATC_DS'] 
        G.add_edge(atc_cs, atc_ds)

   
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    plt.title(f'Árbol de recorrido del Tren {tren} el {fecha}')
    plt.show()

def comparar_consumo_trenes(df, columna_asset, columna_fecha, columna_energia):
    """
    Compara el consumo de energía entre trenes usando un tren base aleatorio.

    Parámetros:
    df: DataFrame con los datos
    columna_asset: nombre de la columna que contiene los identificadores de los trenes
    columna_fecha: nombre de la columna que contiene las fechas
    columna_energia: nombre de la columna que contiene los valores de energía

    Retorna:
    DataFrame con los resultados de las comparaciones
    """
    # Configurar la semilla aleatoria
    random.seed(2024)

    # Obtener la lista de trenes excluyendo el primero
    opciones = df[columna_asset].value_counts()[1:].index.tolist()

    # Seleccionar un tren base aleatorio
    tren_base = random.choice(opciones)

    # Agrupar por fecha y asset, obtener el máximo
    df_consumo = df.groupby([columna_asset, columna_fecha])[columna_energia].max().reset_index()

    # Obtener los datos del tren base
    datos_tren_base = df_consumo[df_consumo[columna_asset] == tren_base][columna_energia]

    # Lista para almacenar resultados
    resultados = []

    # Realizar t-test para cada tren contra el tren base
    for tren in df_consumo[columna_asset].unique():
        if tren != tren_base:
            datos_tren = df_consumo[df_consumo[columna_asset] == tren][columna_energia]

            # Realizar t-test
            t_stat, p_valor = stats.ttest_ind(datos_tren_base, datos_tren)

            # Calcular estadísticas descriptivas
            media_base = datos_tren_base.mean()
            media_tren = datos_tren.mean()
            diferencia_medias = media_tren - media_base

            resultados.append({
                'tren_comparado': tren,
                'tren_base': tren_base,
                'media_tren_base': media_base,
                'media_tren_comparado': media_tren,
                'diferencia_medias': diferencia_medias,
                'estadistico_t': t_stat,
                'p_valor': p_valor,
                'significativo': p_valor < 0.05
            })

    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)

    return df_resultados, tren_base

def test_codo(X, max_clusters=10, random_state=42):
    """
    Función que evalúa diferentes métricas para seleccionar el número óptimo de clústeres.

    Parámetros:
    - X: matriz de características escaladas.
    - max_clusters: número máximo de clústeres a evaluar.
    - random_state: semilla para la aleatoriedad de KMeans.

    Retorna:
    - Un gráfico con las métricas evaluadas: Calinski-Harabasz y Davies-Bouldin.
    """


    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []


    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++')
        clusters = kmeans.fit_predict(X)


        inertias.append(kmeans.inertia_)

        # Mayor, Mejor
        calinski = calinski_harabasz_score(X, clusters)
        calinski_scores.append(calinski)

        #  menor, mejor
        davies_bouldin = davies_bouldin_score(X, clusters)
        davies_bouldin_scores.append(davies_bouldin)


    fig, ax1 = plt.subplots(figsize=(10, 6))


    ax1.plot(range(2, max_clusters+1), calinski_scores, 'b-', label="Calinski-Harabasz")
    ax1.set_xlabel("Número de Clústeres")
    ax1.set_ylabel("Calinski-Harabasz", color='b')

    ax2 = ax1.twinx()


    ax2.plot(range(2, max_clusters+1), davies_bouldin_scores, 'r-', label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin", color='r')


    plt.title("Evaluación del Número de Clústeres")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

    # Retornar las métricas calculadas
    return calinski_scores, davies_bouldin_scores, inertias