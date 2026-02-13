import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Визуализация графов", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("📊 Визуализация графов на основе корреляционной матрицы")

# Создание вкладок
tab1, tab2, tab3 = st.tabs(["📈 Визуализация", "📋 Данные", "ℹ️ О программе"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Параметры")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "📁 Выберите Excel файл",
            type=['xlsx', 'xls'],
            help="Загрузите файл с данными для анализа"
        )
        
        # Выбор типа графа
        graph_type = st.radio(
            "📐 Тип визуализации",
            ["2D", "3D"],
            horizontal=True,
            help="Выберите тип отображения графа"
        )
        
        # Дополнительные параметры
        st.subheader("🎨 Настройки отображения")
        show_labels = st.checkbox("Показывать подписи узлов", value=True)
        show_weights = st.checkbox("Показывать веса связей", value=True)
        
        # Фильтр по корреляции
        min_correlation = st.slider(
            "🔍 Минимальная корреляция для отображения",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Отображать только связи с корреляцией выше указанного значения"
        )
        
        # Кнопка построения
        plot_button = st.button(
            "🚀 Построить граф", 
            type="primary", 
            use_container_width=True
        )
        
        # Отображение изображения
        if os.path.exists('example.png'):
            st.image('example.png', caption='Пример оформления', use_container_width=True)
    
    with col2:
        st.subheader("📊 Результат визуализации")
        
        if uploaded_file is not None:
            try:
                # Чтение данных
                data = pd.read_excel(uploaded_file)
                st.success(f"✅ Загружено {len(data)} строк и {len(data.columns)} столбцов")
                
                # Показываем превью данных
                with st.expander("👁️ Превью данных"):
                    st.dataframe(data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла: {str(e)}")
                data = None
        else:
            st.info("👆 Загрузите Excel файл для начала работы")
            data = None
        
        # Область для графика
        plot_placeholder = st.empty()
        
        if plot_button and uploaded_file is not None and data is not None:
            try:
                with st.spinner("🔄 Построение графа..."):
                    # Очищаем предыдущие графики
                    plt.close('all')
                    
                    # Подготовка данных
                    data_array = data.select_dtypes(include=[np.number]).to_numpy()
                    
                    if data_array.size == 0:
                        st.warning("⚠️ В файле нет числовых данных для анализа")
                    else:
                        # Вычисление корреляционной матрицы
                        corr_matrix = np.corrcoef(data_array.T)
                        corr_matrix = np.round(corr_matrix, 2)
                        
                        # Создание графа
                        G = nx.Graph()
                        column_names = data.select_dtypes(include=[np.number]).columns.tolist()
                        
                        # Добавление узлов
                        for i, name in enumerate(column_names):
                            G.add_node(i, label=name)
                        
                        # Добавление ребер с фильтром
                        n = len(column_names)
                        for i in range(n):
                            for j in range(i+1, n):
                                weight = corr_matrix[i, j]
                                if not np.isnan(weight) and abs(weight) >= min_correlation:
                                    G.add_edge(i, j, weight=weight)
                        
                        if G.number_of_edges() == 0:
                            st.warning(f"⚠️ Нет связей с корреляцией >= {min_correlation}. Уменьшите порог.")
                        else:
                            # Создание фигуры
                            fig = plt.figure(figsize=(12, 8))
                            
                            if graph_type == "2D":
                                ax = fig.add_subplot(111)
                                pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
                                
                                # Рисование узлов
                                nx.draw_networkx_nodes(G, pos, ax=ax, 
                                                      node_color='lightblue',
                                                      node_size=800, 
                                                      alpha=0.8,
                                                      edgecolors='darkblue',
                                                      linewidths=2)
                                
                                # Подготовка ребер
                                edges = G.edges()
                                if edges:
                                    weights = [abs(G[u][v]['weight']) for u, v in edges]
                                    max_weight = max(weights) if weights else 1
                                    
                                    # Цвет и толщина в зависимости от корреляции
                                    for u, v in edges:
                                        weight = G[u][v]['weight']
                                        width = 1 + 3 * abs(weight) / max_weight
                                        color = 'red' if weight > 0 else 'blue'
                                        
                                        nx.draw_networkx_edges(G, pos, ax=ax,
                                                              edgelist=[(u, v)],
                                                              width=width,
                                                              edge_color=color,
                                                              alpha=0.6)
                                
                                # Подписи узлов
                                if show_labels:
                                    labels = {i: column_names[i] for i in G.nodes()}
                                    nx.draw_networkx_labels(G, pos, labels, ax=ax, 
                                                           font_size=9, 
                                                           font_weight='bold')
                                
                                # Подписи ребер
                                if show_weights and edges:
                                    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                                                  for u, v in edges}
                                    nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                                                ax=ax, font_size=8)
                                
                                ax.set_title(f"2D визуализация графа\n"
                                           f"Узлов: {G.number_of_nodes()}, Связей: {G.number_of_edges()}", 
                                           fontsize=14, fontweight='bold')
                                ax.axis('off')
                                
                            else:  # 3D
                                ax = fig.add_subplot(111, projection='3d')
                                pos_3d = nx.spring_layout(G, dim=3, seed=42, k=3, iterations=100)
                                
                                # Извлечение координат
                                xs = [pos_3d[node][0] for node in G.nodes()]
                                ys = [pos_3d[node][1] for node in G.nodes()]
                                zs = [pos_3d[node][2] for node in G.nodes()]
                                
                                # Рисование узлов
                                ax.scatter(xs, ys, zs, c='lightblue', s=200, 
                                          alpha=0.8, edgecolors='darkblue', linewidth=2)
                                
                                # Рисование ребер
                                for edge in G.edges():
                                    x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
                                    y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
                                    z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
                                    
                                    weight = G[edge[0]][edge[1]]['weight']
                                    color = 'red' if weight > 0 else 'blue'
                                    linewidth = 1 + 3 * abs(weight)
                                    
                                    ax.plot(x, y, z, color=color, alpha=0.6, 
                                           linewidth=linewidth)
                                
                                # Подписи узлов
                                if show_labels:
                                    for i, node in enumerate(G.nodes()):
                                        ax.text(pos_3d[node][0], pos_3d[node][1], pos_3d[node][2], 
                                               column_names[node], fontsize=9, fontweight='bold')
                                
                                ax.set_title(f"3D визуализация графа\n"
                                           f"Узлов: {G.number_of_nodes()}, Связей: {G.number_of_edges()}", 
                                           fontsize=14, fontweight='bold')
                                ax.set_xlabel('X')
                                ax.set_ylabel('Y')
                                ax.set_zlabel('Z')
                            
                            plt.tight_layout()
                            
                            # Отображаем график
                            plot_placeholder.pyplot(fig)
                            
                            # Статистика в колонках
                            st.subheader("📊 Статистика")
                            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                            
                            with col_stats1:
                                st.metric("Количество узлов", G.number_of_nodes())
                            with col_stats2:
                                st.metric("Количество связей", G.number_of_edges())
                            with col_stats3:
                                if G.edges():
                                    avg_weight = np.mean([abs(G[u][v]['weight']) 
                                                         for u, v in G.edges()])
                                    st.metric("Средняя корреляция", f"{avg_weight:.3f}")
                                else:
                                    st.metric("Средняя корреляция", "N/A")
                            with col_stats4:
                                density = nx.density(G)
                                st.metric("Плотность графа", f"{density:.3f}")
                            
                            # Информация о корреляциях
                            with st.expander("📈 Матрица корреляций"):
                                corr_df = pd.DataFrame(corr_matrix, 
                                                      index=column_names, 
                                                      columns=column_names)
                                st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'), 
                                           use_container_width=True)
                            
            except Exception as e:
                st.error(f"❌ Ошибка построения графа: {str(e)}")
                st.exception(e)

with tab2:
    st.subheader("📋 Загруженные данные")
    if uploaded_file is not None and data is not None:
        st.dataframe(data, use_container_width=True)
        
        # Статистика по данным
        st.subheader("📊 Описательная статистика")
        st.dataframe(data.describe(), use_container_width=True)
    else:
        st.info("Загрузите файл для просмотра данных")

with tab3:
    st.header("ℹ️ О приложении")
    st.markdown("""
    ### Визуализация графов на основе корреляционной матрицы
    
    **Функциональность:**
    - 📁 Загрузка данных из Excel файлов
    - 📊 Автоматическое вычисление корреляционной матрицы
    - 📈 Построение графа в 2D и 3D
    - 🎨 Настройка порога корреляции
    - 📍 Отображение подписей и весов
    
    **Формат данных:**
    - Файл Excel должен содержать числовые данные
    - Каждый столбец - отдельная переменная
    - Строки - наблюдения
    
    **Интерпретация:**
    - 🔴 Красные линии - положительная корреляция
    - 🔵 Синие линии - отрицательная корреляция
    - 📏 Толщина линии пропорциональна силе связи
    
    **Технологии:**
    - 🚀 Streamlit для веб-интерфейса
    - 🔗 NetworkX для работы с графами
    - 📐 Matplotlib для визуализации
    - 🐼 Pandas для обработки данных
    """)

# Добавляем footer
st.markdown("---")
st.markdown("👨‍💻 Разработано с использованием Streamlit, NetworkX и Matplotlib")
