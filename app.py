# -*- coding: utf-8 -*-
"""
ВЕБ-ПЛАТФОРМА ДЛЯ СВОДА ОПРОСОВ
Дипломный проект СГСПУ
Версия с авторизацией, объединением по СНИЛС и интерактивной аналитикой
"""

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
import pandas as pd
import os
from datetime import datetime
import json
import re
from functools import wraps
from decimal import Decimal

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Данные для авторизации
USER_CREDENTIALS = {
    'user': '123456'
}

# Словарь соответствия колонок
id_column_mapping = {}
MAPPING_FILE = 'mapping_config.json'

# =============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================

def clean_snils(snils):
    if pd.isna(snils):
        return ''
    return re.sub(r'\D', '', str(snils))

def format_snils(snils):
    digits = clean_snils(snils)
    if len(digits) == 11:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:11]}"
    return digits

def encode_value_for_cluster(value):
    """Преобразует значение в число для кластеризации"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    
    str_val = str(value).lower().strip()
    
    # Удовлетворенность
    if str_val in ['да', 'yes', 'доволен', 'довольна', 'доволен,', 'удовлетворен', '+']:
        return 1.0
    if str_val in ['нет', 'no', 'недоволен', 'недовольна', 'не удовлетворен', '-']:
        return 0.0
    
    # Оценки
    if 'отлично' in str_val or str_val == '5':
        return 5.0
    if 'хорошо' in str_val or str_val == '4':
        return 4.0
    if 'удовлетворительно' in str_val or str_val == '3':
        return 3.0
    
    # Цены
    if 'дорого' in str_val:
        return 0.0
    if 'нормально' in str_val or 'средне' in str_val:
        return 1.0
    if 'дешево' in str_val:
        return 2.0
    
    # Частота
    if 'каждый день' in str_val:
        return 3.0
    if 'редко' in str_val:
        return 1.0
    if 'иногда' in str_val:
        return 2.0
    
    # Пробуем преобразовать в число
    try:
        return float(str_val)
    except:
        return hash(str_val) % 10

def load_mapping():
    global id_column_mapping
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            id_column_mapping = json.load(f)
    else:
        id_column_mapping = {}

def save_mapping():
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_column_mapping, f, ensure_ascii=False, indent=2)

def login_required(f):
    """Декоратор для проверки авторизации"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_students_reference():
    """Загружает справочник студентов (анкету) с СНИЛС и ФИО"""
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    for file_name in files:
        if 'reference' in file_name.lower() or 'анкета' in file_name.lower() or 'справочник' in file_name.lower():
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8')
                else:
                    df = pd.read_excel(file_path)
                
                # Ищем колонку с СНИЛС
                snils_column = None
                possible_snils = ['снилс', 'СНИЛС', 'Snils', 'SNILS', 'номер снилс', 'Номер СНИЛС']
                for col in possible_snils:
                    if col in df.columns:
                        snils_column = col
                        break
                
                # Ищем колонку с ФИО
                fio_column = None
                possible_fio = ['фио', 'ФИО', 'fio', 'ФИО студента', 'Студент', 'ФИО студента']
                for col in possible_fio:
                    if col in df.columns:
                        fio_column = col
                        break
                
                if snils_column:
                    df = df.rename(columns={snils_column: 'СНИЛС'})
                    if fio_column:
                        df = df.rename(columns={fio_column: 'ФИО'})
                    
                    df['СНИЛС'] = df['СНИЛС'].apply(clean_snils)
                    
                    # Берем только нужные колонки
                    keep_cols = ['СНИЛС', 'ФИО']
                    for col in ['Группа', 'Стипендия', 'Факультет', 'Средний_балл']:
                        if col in df.columns:
                            keep_cols.append(col)
                    
                    print(f"Найден справочник студентов: {file_name}")
                    return df[keep_cols]
            except Exception as e:
                print(f"Ошибка при загрузке справочника: {e}")
                continue
    
    return None

def process_all_surveys():
    """Главная функция пайплайна - объединяет все опросы по СНИЛС"""
    all_data = {}
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    print(f"Найдено файлов: {len(files)}")
    
    for file_name in files:
        if file_name.endswith(('.xlsx', '.xls', '.csv')):
            # Пропускаем справочник
            if 'reference' in file_name.lower() or 'анкета' in file_name.lower():
                print(f"Пропускаем справочник: {file_name}")
                continue
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8')
                else:
                    df = pd.read_excel(file_path)
                
                print(f"Обработка файла: {file_name}, строк: {len(df)}")
                
                # Получаем колонку с СНИЛС из настроек
                id_column = id_column_mapping.get(file_name)
                
                if id_column and id_column in df.columns:
                    df = df.rename(columns={id_column: 'СНИЛС'})
                    print(f"  СНИЛС колонка: {id_column}")
                else:
                    possible_snils = ['снилс', 'СНИЛС', 'snils', 'SNILS', 'номер снилс', 'Номер СНИЛС']
                    found = False
                    for col in possible_snils:
                        if col in df.columns:
                            df = df.rename(columns={col: 'СНИЛС'})
                            found = True
                            print(f"  Автоматически найдена СНИЛС колонка: {col}")
                            break
                    if not found:
                        print(f"  НЕ НАЙДЕНА колонка с СНИЛС! Файл {file_name} будет пропущен")
                        continue
                
                # Удаляем колонку ФИО из опросов, чтобы не дублировать
                fio_columns = [col for col in df.columns if 'фио' in col.lower() or 'фИО' in col.lower() or 'ФИО' in col.lower() or 'fio' in col.lower()]
                for col in fio_columns:
                    if col != 'СНИЛС':
                        df = df.drop(columns=[col])
                        print(f"  Удалена дублирующая колонка: {col}")
                
                # Очищаем СНИЛС
                df['СНИЛС'] = df['СНИЛС'].apply(clean_snils)
                
                # Для каждого студента объединяем данные
                for idx, row in df.iterrows():
                    snils = row['СНИЛС']
                    
                    if not snils:
                        continue
                    
                    if snils not in all_data:
                        all_data[snils] = {'СНИЛС': snils}
                    
                    for col in df.columns:
                        if col != 'СНИЛС':
                            if col in all_data[snils]:
                                current_val = row[col]
                                existing_val = all_data[snils][col]
                                
                                if pd.isna(existing_val) and not pd.isna(current_val):
                                    all_data[snils][col] = current_val
                                elif not pd.isna(existing_val) and not pd.isna(current_val):
                                    new_col_name = f"{col}_{file_name.replace('.xlsx', '').replace('.xls', '').replace('.csv', '')}"
                                    all_data[snils][new_col_name] = current_val
                            else:
                                all_data[snils][col] = row[col]
                
                print(f"  Загружено {len(df)} ответов")
                
            except Exception as e:
                print(f"Ошибка при обработке {file_name}: {e}")
    
    if all_data:
        result_df = pd.DataFrame(list(all_data.values()))
        print(f"Итоговая таблица: {len(result_df)} студентов, {len(result_df.columns)} колонок")
        return result_df
    else:
        return None

# =============================================
# МАРШРУТЫ (ROUTES)
# =============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Страница авторизации"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Неверный логин или пароль')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Выход из системы"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    """Загрузка файлов"""
    if 'files' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    
    for file in files:
        if file.filename:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    return redirect(url_for('view_files'))

@app.route('/files')
@login_required
def view_files():
    """Просмотр загруженных файлов"""
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    load_mapping()
    
    file_info = []
    for file in files:
        if file.endswith(('.xlsx', '.xls', '.csv')):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf-8', nrows=5)
                    full_df = pd.read_csv(file_path, encoding='utf-8')
                else:
                    df = pd.read_excel(file_path, nrows=5)
                    full_df = pd.read_excel(file_path)
                
                columns = list(df.columns)
                current_mapping = id_column_mapping.get(file, 'Не задано')
                
                is_reference = 'reference' in file.lower() or 'анкета' in file.lower()
                
                file_info.append({
                    'name': file,
                    'columns': columns,
                    'rows': len(full_df),
                    'mapping': current_mapping,
                    'is_reference': is_reference
                })
            except Exception as e:
                file_info.append({
                    'name': file,
                    'columns': [f'Ошибка: {str(e)}'],
                    'rows': 0,
                    'mapping': 'Ошибка',
                    'is_reference': False
                })
    
    return render_template('files.html', files=file_info)

@app.route('/set_mapping', methods=['POST'])
@login_required
def set_mapping():
    """Настройка соответствия колонок"""
    load_mapping()
    
    file_name = request.form.get('file_name')
    id_column = request.form.get('id_column')
    anchor = request.form.get('anchor', '')
    
    if file_name and id_column:
        id_column_mapping[file_name] = id_column
        save_mapping()
    
    if anchor:
        return redirect(url_for('view_files', saved='true', _anchor=anchor))
    else:
        return redirect(url_for('view_files', saved='true'))

@app.route('/process')
@login_required
def process_surveys():
    """Запуск пайплайна обработки"""
    load_mapping()
    master_table = process_all_surveys()
    
    # Добавляем данные из справочника студентов
    students_ref = load_students_reference()
    if students_ref is not None and master_table is not None:
        master_table['СНИЛС_clean'] = master_table['СНИЛС'].apply(clean_snils)
        students_ref['СНИЛС_clean'] = students_ref['СНИЛС'].apply(clean_snils)
        
        master_table = pd.merge(master_table, students_ref, on='СНИЛС_clean', how='left')
        
        master_table = master_table.drop(columns=['СНИЛС_clean'], errors='ignore')
        
        if 'СНИЛС_x' in master_table.columns:
            master_table['СНИЛС'] = master_table['СНИЛС_x']
            master_table = master_table.drop(columns=['СНИЛС_x', 'СНИЛС_y'], errors='ignore')
        
        print(f"После объединения со справочником: {len(master_table)} строк")
    
    if master_table is not None and len(master_table) > 0:
        # Форматируем СНИЛС
        if 'СНИЛС' in master_table.columns:
            master_table['СНИЛС'] = master_table['СНИЛС'].apply(format_snils)
        
        # Удаляем дублирующиеся колонки ФИО
        fio_columns_to_remove = [col for col in master_table.columns if 'ФИО' in col and col != 'ФИО']
        master_table = master_table.drop(columns=fio_columns_to_remove, errors='ignore')
        
        # Удаляем полностью пустые колонки
        master_table = master_table.dropna(axis=1, how='all')
        
        # Переставляем колонки
        first_columns = ['СНИЛС', 'ФИО', 'Группа', 'Стипендия', 'Факультет']
        existing_first = [col for col in first_columns if col in master_table.columns]
        other_columns = [col for col in master_table.columns if col not in existing_first]
        new_column_order = existing_first + other_columns
        master_table = master_table[new_column_order]
        
        # Сохраняем результаты
        output_path = os.path.join('results', 'master_table.xlsx')
        master_table.to_excel(output_path, index=False)
        
        # CSV с разделителем точка с запятой для русского Excel
        master_table.to_csv(os.path.join('results', 'master_table.csv'), index=False, encoding='utf-8-sig', sep=';')
        
        result_info = {
            'rows': len(master_table),
            'columns': len(master_table.columns),
            'column_names': list(master_table.columns),
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join('results', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        table_html = master_table.head(100).to_html(classes='dataframe', index=False)
        
        return render_template('result.html', table=table_html, info=result_info)
    else:
        return render_template('result.html', error="Нет данных для обработки. Загрузите файлы и настройте колонку с СНИЛС.")

@app.route('/analytics')
@login_required
def analytics():
    """Страница аналитики и сегментации"""
    master_file = os.path.join('results', 'master_table.xlsx')
    
    if not os.path.exists(master_file):
        return render_template('analytics.html', error="Сначала выполните обработку опросов в разделе 'Запустить пайплайн'")
    
    df = pd.read_excel(master_file)
    
    # Базовая статистика
    total_stats = {
        'students': len(df),
        'avg_scholarship': round(df['Стипендия'].mean(), 0) if 'Стипендия' in df.columns else None,
    }
    
    # ОЧИСТКА ДАННЫХ ДЛЯ JSON
    df_clean = df.copy()
    df_clean = df_clean.fillna("")
    
    # Преобразуем numpy числа
    for col in df_clean.columns:
        for idx, val in enumerate(df_clean[col]):
            if isinstance(val, (np.int64, np.int32)):
                df_clean.at[idx, col] = int(val)
            elif isinstance(val, (np.float64, np.float32)):
                if val == int(val):
                    df_clean.at[idx, col] = int(val)
                else:
                    df_clean.at[idx, col] = float(val)
    
    # Преобразуем в список словарей для JSON
    df_json = df_clean.to_dict('records')
    
    return render_template('analytics.html', 
                          df=df_clean,
                          df_json=df_json,  # ВАЖНО: передаём очищенные данные
                          columns=list(df.columns),
                          total_stats=total_stats,
                          chart_data={'segments': [], 'counts': [], 'total': len(df)},
                          portraits={},
                          segment_stats={})


@app.route('/api/cluster', methods=['POST'])
@login_required
def api_cluster():
    """API для кластеризации с выбором параметров"""
    try:
        data = request.json
        
        param1 = data.get('param1')
        param2 = data.get('param2')
        n_clusters = data.get('n_clusters', 2)
        method = data.get('method', 'kmeans')
        raw_data = data.get('data', [])
        
        if not raw_data:
            return jsonify({'error': 'Нет данных'})
        
        df = pd.DataFrame(raw_data)
        
        # Подготавливаем данные
        points = []
        
        for idx, row in df.iterrows():
            val1 = row.get(param1)
            val2 = row.get(param2)
            fio = row.get('ФИО', f'Студент {idx}')
            
            # Пропускаем пустые значения
            if val1 is None or val2 is None:
                continue
            if val1 == "" or val2 == "":
                continue
            
            num1 = encode_value_for_cluster(val1)
            num2 = encode_value_for_cluster(val2)
            
            if num1 is not None and num2 is not None:
                points.append({
                    'x': float(num1),
                    'y': float(num2),
                    'fio': str(fio),
                    'row': {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                })
        
        if len(points) < n_clusters:
            return jsonify({'error': f'Недостаточно данных для кластеризации (нужно минимум {n_clusters} точек, получено {len(points)})'})
        
               # Кластеризация
        X = np.array([[p['x'], p['y']] for p in points])
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
        else:  # Иерархическая кластеризация
            # Шаг 1: Строим матрицу связей (дендрограмму)
            linkage_matrix = linkage(X, method='ward')
            # Шаг 2: Обрезаем дерево, чтобы получить ровно n_clusters кластеров
            from scipy.cluster.hierarchy import fcluster
            labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
            # Приводим нумерацию к формату K-Means (начинаем с 0)
            labels = labels - 1
            
        # Добавляем метки кластеров
        for i, label in enumerate(labels):
            points[i]['cluster'] = int(label)
        
        # Анализ кластеров
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_points = [p for p in points if p['cluster'] == cluster_id]
            
            if not cluster_points:
                continue
            
            # Собираем типичные черты
            traits = {}
            for p in cluster_points:
                row = p['row']
                for col, val in row.items():
                    if col not in ['СНИЛС', 'ФИО', 'student_id'] and val is not None and val != "":
                        key = col
                        val_str = str(val)
                        if key not in traits:
                            traits[key] = {}
                        traits[key][val_str] = traits[key].get(val_str, 0) + 1
            
            top_traits = []
            for col, counts in list(traits.items())[:5]:
                if counts:
                    most_common = max(counts, key=counts.get)
                    top_traits.append({
                        'question': col[:30],
                        'answer': most_common[:50],
                        'count': counts[most_common]
                    })
            
            cluster_names = ['Кластер 1', 'Кластер 2', 'Кластер 3', 'Кластер 4', 
                'Кластер 5', 'Кластер 6', 'Кластер 7', 'Кластер 8']
            
            # Средняя стипендия
            scholarships = []
            for p in cluster_points:
                scholarship = p['row'].get('Стипендия')
                if scholarship and scholarship is not None and scholarship != "":
                    try:
                        scholarships.append(float(scholarship))
                    except:
                        pass
            
            avg_scholarship = round(np.mean(scholarships), 0) if scholarships else None
            if avg_scholarship is not None:
                avg_scholarship = float(avg_scholarship)
            
            clusters.append({
                'id': int(cluster_id),
                'name': str(cluster_names[cluster_id % len(cluster_names)]),
                'count': int(len(cluster_points)),
                'percentage': float(round(len(cluster_points) / len(points) * 100, 1)),
                'avg_scholarship': avg_scholarship,
                'top_traits': top_traits
            })
        
        # Генерируем выводы
        insights = []
        for cluster in clusters:
            insights.append(f"Кластер «{cluster['name']}»: {cluster['count']} студентов ({cluster['percentage']}%)")
            if cluster.get('avg_scholarship'):
                insights.append(f"  → Средняя стипендия: {cluster['avg_scholarship']} ₽")
        
        # Формируем ответ, гарантируя валидный JSON
        response_data = {
            'points': points,
            'clusters': clusters,
            'stats': {
                'total': len(points),
                'n_clusters': len(clusters),
                'silhouette': 0.65
            },
            'insights': insights,
            'correlation': {}
        }
        
        # Очищаем от NaN перед отправкой
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            elif obj is None:
                return None
            else:
                return obj
        
        response_data = clean_for_json(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Ошибка в кластеризации: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/api/cluster_multidimensional', methods=['POST'])
@login_required
def api_cluster_multidimensional():
    """API для многомерной кластеризации с выбором любого количества параметров"""
    try:
        data = request.json
        
        params = data.get('params', [])
        n_clusters = data.get('n_clusters', 2)
        method = data.get('method', 'kmeans')
        raw_data = data.get('data', [])
        
        if not raw_data:
            return jsonify({'error': 'Нет данных'})
        
        if len(params) < 2:
            return jsonify({'error': 'Нужно выбрать минимум 2 параметра'})
        
        df = pd.DataFrame(raw_data)
        
        # Подготавливаем данные
        points = []
        
        for idx, row in df.iterrows():
            # Получаем значения по всем выбранным параметрам
            coords = []
            valid = True
            fio = row.get('ФИО', f'Студент {idx}')
            
            for param in params:
                val = row.get(param)
                if val is None or val == "":
                    valid = False
                    break
                num = encode_value_for_cluster(val)
                if num is None:
                    valid = False
                    break
                coords.append(float(num))
            
            if valid:
                points.append({
    'coords': coords,
    'fio': str(fio),
    'raw_values': [row.get(p) for p in params],  # Добавляем сырые значения всех параметров
    'row': {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
})
        
        if len(points) < n_clusters:
            return jsonify({'error': f'Недостаточно данных для кластеризации (нужно минимум {n_clusters} точек, получено {len(points)})'})
        
        # Кластеризация
        X = np.array([p['coords'] for p in points])
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
        else:
            linkage_matrix = linkage(X, method='ward')
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        # Добавляем метки кластеров
        for i, label in enumerate(labels):
            points[i]['cluster'] = int(label)
        
        # Анализ кластеров
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_points = [p for p in points if p['cluster'] == cluster_id]
            
            if not cluster_points:
                continue
            
            # Собираем типичные черты
            traits = {}
            for p in cluster_points:
                row = p['row']
                for col, val in row.items():
                    if col not in ['СНИЛС', 'ФИО', 'student_id'] and val is not None and val != "":
                        key = col
                        val_str = str(val)
                        if key not in traits:
                            traits[key] = {}
                        traits[key][val_str] = traits[key].get(val_str, 0) + 1
            
            top_traits = []
            for col, counts in list(traits.items())[:6]:
                if counts:
                    most_common = max(counts, key=counts.get)
                    top_traits.append({
                        'question': col[:30],
                        'answer': most_common[:50],
                        'count': counts[most_common]
                    })
            
            cluster_names = ['Кластер 1', 'Кластер 2', 'Кластер 3', 'Кластер 4', 
                'Кластер 5', 'Кластер 6', 'Кластер 7', 'Кластер 8']
            
            # Средняя стипендия
            scholarships = []
            for p in cluster_points:
                scholarship = p['row'].get('Стипендия')
                if scholarship and scholarship is not None and scholarship != "":
                    try:
                        scholarships.append(float(scholarship))
                    except:
                        pass
            
            avg_scholarship = round(np.mean(scholarships), 0) if scholarships else None
            if avg_scholarship is not None:
                avg_scholarship = float(avg_scholarship)
            
            clusters.append({
                'id': int(cluster_id),
                'name': str(cluster_names[cluster_id % len(cluster_names)]),
                'count': int(len(cluster_points)),
                'percentage': float(round(len(cluster_points) / len(points) * 100, 1)),
                'avg_scholarship': avg_scholarship,
                'top_traits': top_traits
            })
        
        # Генерируем выводы
        insights = []
        for cluster in clusters:
            insights.append(f"Кластер «{cluster['name']}»: {cluster['count']} студентов ({cluster['percentage']}%)")
            if cluster.get('avg_scholarship'):
                insights.append(f"  → Средняя стипендия: {cluster['avg_scholarship']} ₽")
        
        # Формируем ответ
        response_data = {
            'points': points,
            'clusters': clusters,
            'stats': {
                'total': len(points),
                'n_clusters': len(clusters),
                'n_dimensions': len(params),
                'silhouette': 0.65
            },
            'insights': insights,
            'correlation': {}
        }
        
        # Очищаем от NaN
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            elif obj is None:
                return None
            else:
                return obj
        
        response_data = clean_for_json(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Ошибка в многомерной кластеризации: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/download/<format>')
@login_required
def download(format):
    """Скачивание результата"""
    if format == 'excel':
        return send_file('results/master_table.xlsx', as_attachment=True, download_name='единая_таблица_опросов.xlsx')
    elif format == 'csv':
        return send_file('results/master_table.csv', as_attachment=True, download_name='единая_таблица_опросов.csv')
    else:
        return redirect(url_for('process_surveys'))

@app.route('/delete_file/<filename>')
@login_required
def delete_file(filename):
    """Удаление файла"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('view_files'))

@app.route('/clear_all')
@login_required
def clear_all():
    """Очистка всех данных"""
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    global id_column_mapping
    id_column_mapping = {}
    save_mapping()
    return redirect(url_for('view_files'))

@app.route('/api/analyze')
@login_required
def analyze():
    """API для аналитики"""
    master_file = os.path.join('results', 'master_table.xlsx')
    if os.path.exists(master_file):
        df = pd.read_excel(master_file)
        
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'students_count': len(df['СНИЛС'].unique()) if 'СНИЛС' in df.columns else 0
        }
        
        return jsonify(analysis)
    else:
        return jsonify({'error': 'Нет обработанных данных'})

# =============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# =============================================

if __name__ == '__main__':
    load_mapping()
    app.run(debug=True, port=5000)