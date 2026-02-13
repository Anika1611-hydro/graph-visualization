import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from PIL import Image, ImageTk
import os

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Визуализация графов")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Переменные
        self.data = None
        self.column_names = None
        
        # Создание вкладок
        self.notebook = ttk.Notebook(root)
        self.notebook.place(x=0, y=0, width=1000, height=690)
        
        # Главная вкладка
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Главная")
        
        self.setup_main_tab()
        
    def setup_main_tab(self):
        """Настройка главной вкладки"""
        
        # Заголовок
        title_label = tk.Label(
            self.main_tab, 
            text="Пример оформления таблицы данных",
            font=("Arial", 12),
            fg="blue"
        )
        title_label.place(x=623, y=573)
        
        # Изображение (если существует)
        self.image_label = tk.Label(self.main_tab)
        self.image_label.place(x=495, y=375, width=500, height=200)
        self.load_image('example.png')
        
        # Графики
        self.create_plot_areas()
        
        # Таблица данных
        self.create_table()
        
        # Кнопки и элементы управления
        self.create_controls()
        
    def load_image(self, image_path):
        """Загрузка и отображение изображения"""
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image = image.resize((500, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")
    
    def create_plot_areas(self):
        """Создание областей для графиков"""
        # Первый график (2D)
        self.fig1, self.ax1 = plt.subplots(figsize=(4.8, 3.5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.main_tab)
        self.canvas1.get_tk_widget().place(x=10, y=10, width=480, height=350)
        
        # Второй график (3D)
        self.fig2 = plt.figure(figsize=(4.8, 3.5))
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.main_tab)
        self.canvas2.get_tk_widget().place(x=500, y=10, width=480, height=350)
    
    def create_table(self):
        """Создание таблицы для отображения данных"""
        # Фрейм для таблицы и скроллбара
        table_frame = tk.Frame(self.main_tab)
        table_frame.place(x=20, y=375, width=480, height=250)
        
        # Таблица
        columns = ('col1', 'col2', 'col3', 'col4')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        # Настройка колонок
        for col in columns:
            self.tree.heading(col, text=f'Колонка {col[3:]}')
            self.tree.column(col, width=100, anchor='center')
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_controls(self):
        """Создание кнопок и элементов управления"""
        # Кнопка выбора данных
        self.load_btn = tk.Button(
            self.main_tab,
            text="Выбрать данные",
            command=self.load_data,
            width=15,
            height=1
        )
        self.load_btn.place(x=20, y=635)
        
        # Метка для типа графа
        type_label = tk.Label(self.main_tab, text="Тип графа")
        type_label.place(x=700, y=635)
        
        # Выпадающий список
        self.graph_type = ttk.Combobox(
            self.main_tab,
            values=['2D', '3D'],
            state='readonly',
            width=10
        )
        self.graph_type.set('2D')
        self.graph_type.place(x=778, y=635)
        
        # Кнопка построения графа
        self.plot_btn = tk.Button(
            self.main_tab,
            text="Построить граф",
            command=self.plot_graph,
            width=15,
            height=1
        )
        self.plot_btn.place(x=525, y=635)
    
    def load_data(self):
        """Загрузка данных из Excel файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл Excel",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Чтение данных
                self.data = pd.read_excel(file_path)
                self.column_names = self.data.columns.tolist()
                
                # Очистка таблицы
                for item in self.tree.get_children():
                    self.tree.delete(item)
                
                # Настройка колонок
                columns = list(self.data.columns)
                self.tree['columns'] = columns
                for col in columns:
                    self.tree.heading(col, text=col)
                    self.tree.column(col, width=100, anchor='center')
                
                # Заполнение данными
                for index, row in self.data.iterrows():
                    values = [str(row[col]) if pd.notna(row[col]) else '' for col in columns]
                    self.tree.insert('', 'end', values=values)
                
                messagebox.showinfo("Успех", f"Загружено {len(self.data)} строк")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def plot_graph(self):
        """Построение графа на основе загруженных данных"""
        if self.data is None or self.data.empty:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
        
        try:
            # Подготовка данных
            data_array = self.data.to_numpy()
            
            # Вычисление корреляционной матрицы
            corr_matrix = np.corrcoef(data_array.T)
            corr_matrix = np.round(corr_matrix, 2)
            
            # Создание графа
            G = nx.Graph()
            
            # Добавление узлов
            for i, name in enumerate(self.column_names):
                G.add_node(i, label=name)
            
            # Добавление ребер (без петель)
            n = len(self.column_names)
            for i in range(n):
                for j in range(i+1, n):
                    weight = corr_matrix[i, j]
                    if not np.isnan(weight):  # Проверка на NaN
                        G.add_edge(i, j, weight=weight)
            
            # Очистка графиков
            self.ax1.clear()
            self.ax2.clear()
            
            # Построение в зависимости от выбранного типа
            if self.graph_type.get() == '2D':
                self.plot_2d_graph(G)
            else:
                self.plot_3d_graph(G)
            
            # Обновление канвасов
            self.canvas1.draw()
            self.canvas2.draw()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить граф: {str(e)}")
    
    def plot_2d_graph(self, G):
        """Построение 2D графа"""
        pos = nx.spring_layout(G, seed=42)
        
        # Рисование узлов
        nx.draw_networkx_nodes(G, pos, ax=self.ax1, node_color='lightblue', 
                               node_size=500, alpha=0.8)
        
        # Рисование ребер с весами
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Нормализация весов для толщины линий
        if weights:
            max_weight = max(abs(w) for w in weights)
            if max_weight > 0:
                widths = [1 + 2 * abs(w)/max_weight for w in weights]
            else:
                widths = [1] * len(weights)
        else:
            widths = [1] * len(edges)
        
        nx.draw_networkx_edges(G, pos, ax=self.ax1, edge_color='red', 
                               width=widths, alpha=0.7)
        
        # Подписи узлов
        labels = {i: self.column_names[i] for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=self.ax1, font_size=8)
        
        # Подписи ребер
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=self.ax1, font_size=6)
        
        self.ax1.set_title("2D визуализация графа")
        self.ax1.axis('off')
    
    def plot_3d_graph(self, G):
        """Построение 3D графа"""
        # 3D позиции узлов
        pos_3d = nx.spring_layout(G, dim=3, seed=42)
        
        # Извлечение координат
        xs = [pos_3d[node][0] for node in G.nodes()]
        ys = [pos_3d[node][1] for node in G.nodes()]
        zs = [pos_3d[node][2] for node in G.nodes()]
        
        # Рисование узлов
        self.ax2.scatter(xs, ys, zs, c='lightblue', s=100, alpha=0.8)
        
        # Рисование ребер
        for edge in G.edges():
            x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
            y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
            z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
            self.ax2.plot(x, y, z, color='red', alpha=0.7, 
                         linewidth=1 + 2 * abs(G[edge[0]][edge[1]]['weight']))
        
        # Подписи узлов
        for i, node in enumerate(G.nodes()):
            self.ax2.text(pos_3d[node][0], pos_3d[node][1], pos_3d[node][2], 
                         self.column_names[node], fontsize=8)
        
        self.ax2.set_title("3D визуализация графа")
        
        # Настройка вида
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')

def main():
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
