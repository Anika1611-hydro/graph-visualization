import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import networkx as nx
import os
import plotly.graph_objects as go
import plotly.express as px

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
        
        # Отображение изображения с подписью
        if os.path.exists('example.png'):
            try:
                st.image('example.png', use_column_width=True)
                # Жирная и крупная подпись под изображением
                st.markdown(
                    "<p style='text-align: center; font-weight: bold; font-size: 18px;'>"
                    "Пример оформления"
                    "</p>", 
                    unsafe_allow_html=True
                )
            except:
                st.image('example.png')
                st.markdown(
                    "<p style='text-align: center; font-weight: bold;'>"
                    "Пример оформления"
                    "</p>", 
                    unsafe_allow_html=True
                )
    
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
                    
                    if graph_type == "2D":
                        # 2D визуализация с matplotlib
                        fig = plt.figure(figsize=(10, 8))
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
                        
                        # Подписи узлов (увеличенный размер)
                        labels = {i: column_names[i] for i in G.nodes()}
                        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
                        
                        # Подписи ребер (увеличенный размер)
                        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)
                        
                        ax.set_title("2D визуализация графа", fontsize=14, fontweight='bold')
                        ax.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:  # 3D с plotly (интерактивный)
                        # Создание позиций для 3D графа
                        pos_3d = nx.spring_layout(G, dim=3, seed=42)
                        
                        # Создание интерактивного 3D графика с plotly
                        fig = go.Figure()
                        
                        # Извлечение координат узлов
                        node_x = []
                        node_y = []
                        node_z = []
                        node_text = []
                        
                        for node in G.nodes():
                            x, y, z = pos_3d[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_z.append(z)
                            node_text.append(column_names[node])
                        
                        # Добавление узлов
                        fig.add_trace(go.Scatter3d(
                            x=node_x, y=node_y, z=node_z,
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                color='lightblue',
                                line=dict(color='darkblue', width=2)
                            ),
                            text=node_text,
                            textposition="top center",
                            textfont=dict(size=12, color='black', family='Arial', weight='bold'),
                            hoverinfo='text',
                            name='Узлы'
                        ))
                        
                        # Добавление ребер
                        for edge in G.edges():
                            x0, y0, z0 = pos_3d[edge[0]]
                            x1, y1, z1 = pos_3d[edge[1]]
                            weight = G[edge[0]][edge[1]]['weight']
                            
                            # Цвет в зависимости от знака корреляции
                            color = 'red' if weight > 0 else 'blue'
                            
                            # Добавление линии ребра
                            fig.add_trace(go.Scatter3d(
                                x=[x0, x1, None], 
                                y=[y0, y1, None], 
                                z=[z0, z1, None],
                                mode='lines',
                                line=dict(
                                    color=color,
                                    width=2 + 5 * abs(weight),
                                ),
                                hoverinfo='text',
                                text=f'Вес: {weight:.2f}',
                                name=f'Ребро: {column_names[edge[0]]}-{column_names[edge[1]]}',
                                showlegend=False
                            ))
                            
                            # Добавление подписи веса посередине ребра
                            mid_x = (x0 + x1) / 2
                            mid_y = (y0 + y1) / 2
                            mid_z = (z0 + z1) / 2
                            
                            fig.add_trace(go.Scatter3d(
                                x=[mid_x], y=[mid_y], z=[mid_z],
                                mode='text',
                                text=[f'{weight:.2f}'],
                                textposition="middle center",
                                textfont=dict(size=10, color='black', family='Arial', weight='bold'),
                                hoverinfo='none',
                                showlegend=False
                            ))
                        
                        # Настройка layout для интерактивности
                        fig.update_layout(
                            title=dict(
                                text="3D визуализация графа",
                                font=dict(size=16, weight='bold')
                            ),
                            scene=dict(
                                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
                                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
                                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=''),
                                bgcolor='white'
                            ),
                            width=800,
                            height=600,
                            showlegend=False,
                            hovermode='closest'
                        )
                        
                        # Добавление возможности вращения мышью
                        config = {'scrollZoom': True, 'displayModeBar': True}
                        
                        # Отображение интерактивного графика
                        st.plotly_chart(fig, use_container_width=True, config=config)
                        
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
