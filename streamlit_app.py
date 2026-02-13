import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import os

# Настройка страницы
st.set_page_config(layout="wide")

# Заголовок
st.title("Визуализация графов")

# Создание вкладок
tab1, tab2 = st.tabs(["Главная", "О программе"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Параметры")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите Excel файл",
            type=['xlsx', 'xls']
        )
        
        # Выбор типа графа
        graph_type = st.radio(
            "Тип графа",
            ["2D", "3D"],
            horizontal=True
        )
        
        # Кнопка построения
        plot_button = st.button("Построить граф", type="primary")
        
        # Отображение изображения
        if os.path.exists('example.png'):
            try:
                st.image('example.png', caption='Пример оформления', use_column_width=True)
            except:
                st.image('example.png', caption='Пример оформления')
    
    with col2:
        st.subheader("Таблица данных")
        
        if uploaded_file is not None:
            try:
                # Чтение данных
                data = pd.read_excel(uploaded_file)
                st.dataframe(data, use_container_width=True, height=300)
                
                # Информация о загрузке
                st.caption(f"Загружено строк: {len(data)}, столбцов: {len(data.columns)}")
                
            except Exception as e:
                st.error(f"Ошибка загрузки файла: {str(e)}")
                data = None
        else:
            st.info("Ожидание загрузки файла...")
            data = None
        
        # Область для графика
        if plot_button and uploaded_file is not None and data is not None:
            try:
                with st.spinner("Построение графа..."):
                    # Очищаем предыдущие графики
                    plt.close('all')
                    
                    # Подготовка данных
                    data_array = data.to_numpy()
                    
                    # Вычисление корреляционной матрицы
                    corr_matrix = np.corrcoef(data_array.T)
                    corr_matrix = np.round(corr_matrix, 2)
                    
                    # Создание графа
                    G = nx.Graph()
                    column_names = data.columns.tolist()
                    
                    # Добавление узлов
                    for i, name in enumerate(column_names):
                        G.add_node(i, label=name)
                    
                    # Добавление ребер (без петель)
                    n = len(column_names)
                    for i in range(n):
                        for j in range(i+1, n):
                            weight = corr_matrix[i, j]
                            if not np.isnan(weight):
                                G.add_edge(i, j, weight=weight)
                    
                    # Создание фигуры
                    fig = plt.figure(figsize=(10, 8))
                    
                    if graph_type == "2D":
                        ax = fig.add_subplot(111)
                        # Убираем тики
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        pos = nx.spring_layout(G, seed=42)
                        
                        # Рисование узлов
                        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                                              node_size=500, alpha=0.8)
                        
                        # Рисование ребер
                        edges = G.edges()
                        if edges:
                            weights = [abs(G[u][v]['weight']) for u, v in edges]
                            max_weight = max(weights) if weights else 1
                            widths = [1 + 1.5 * w/max_weight for w in weights]
                            
                            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='r',
                                                  width=widths, alpha=0.7)
                        
                        # Подписи узлов
                        labels = {i: column_names[i] for i in G.nodes()}
                        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
                        
                        # Подписи ребер
                        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6)
                        
                        ax.set_title("2D визуализация графа")
                        ax.axis('off')
                        
                    else:  # 3D
                        ax = fig.add_subplot(111, projection='3d')
                        # Убираем тики
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_zticks([])
                        
                        pos_3d = nx.spring_layout(G, dim=3, seed=42)
                        
                        # Извлечение координат
                        xs = [pos_3d[node][0] for node in G.nodes()]
                        ys = [pos_3d[node][1] for node in G.nodes()]
                        zs = [pos_3d[node][2] for node in G.nodes()]
                        
                        # Рисование узлов
                        ax.scatter(xs, ys, zs, c='lightblue', s=100, alpha=0.8)
                        
                        # Рисование ребер
                        for edge in G.edges():
                            x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
                            y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
                            z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
                            
                            weight = abs(G[edge[0]][edge[1]]['weight'])
                            ax.plot(x, y, z, color='r', alpha=0.7, 
                                   linewidth=1 + 2 * weight)
                        
                        # Подписи узлов
                        for i, node in enumerate(G.nodes()):
                            ax.text(pos_3d[node][0], pos_3d[node][1], pos_3d[node][2], 
                                   column_names[node], fontsize=8)
                        
                        ax.set_title("3D визуализация графа")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Ошибка построения графа: {str(e)}")

with tab2:
    st.header("О программе")
    st.markdown("""
    ### Визуализация графов на основе корреляционной матрицы
    
    **Функциональность:**
    - Загрузка данных из Excel файлов
    - Автоматическое вычисление корреляционной матрицы
    - Построение графа в 2D и 3D
    - Визуализация весов связей
    
    **Формат данных:**
    - Файл Excel с числовыми данными
    - Каждый столбец - отдельная переменная
    - Строки - наблюдения
    """)
