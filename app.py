"""
Главный файл запуска ML Platform с веб-интерфейсом
"""
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid

# ============ НАСТРОЙКА ПУТЕЙ ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Рабочая директория: {BASE_DIR}")

app = FastAPI(
    title="ML Platform",
    description="Платформа для создания и тестирования алгоритмов машинного обучения",
    version="1.0.0"
)

# Создаем необходимые директории
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Подключаем статические файлы и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============ МОДЕЛИ ДАННЫХ ============
class User:
    def __init__(self, name: str, email: str, role: str = "Data Scientist"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.email = email
        self.role = role
        self.created_at = datetime.now()

class Project:
    def __init__(self, name: str, description: str, owner: User):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.owner = owner
        self.status = "active"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.experiments = []
        self.tags = []

class Experiment:
    def __init__(self, name: str, algorithm: str, dataset: str, project_id: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.algorithm = algorithm
        self.dataset = dataset
        self.project_id = project_id
        self.status = "created"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.metrics = {}
        self.hyperparameters = {}
        self.artifact_path = None

class TrainedModel:
    def __init__(self, name: str, description: str, experiment_id: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.experiment_id = experiment_id
        self.status = "development"
        self.version = "1.0.0"
        self.created_at = datetime.now()
        self.metrics = {}
        self.deployment_status = None

# ============ ХРАНИЛИЩЕ ДАННЫХ ============
class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        """Инициализация базы данных с демо-данными"""
        # Создаем демо-пользователя
        self.demo_user = User("Алексей Петров", "alexey@mlplatform.com", "Data Scientist")
        
        # Создаем демо-проекты
        self.projects = [
            Project(
                name="Прогнозирование оттока клиентов",
                description="ML модель для предсказания ухода клиентов банка",
                owner=self.demo_user
            ),
            Project(
                name="Обнаружение мошеннических транзакций",
                description="Система для выявления мошенничества в реальном времени",
                owner=self.demo_user
            ),
            Project(
                name="Рекомендательная система товаров",
                description="Персонализированные рекомендации для интернет-магазина",
                owner=self.demo_user
            )
        ]
        
        # Создаем демо-эксперименты
        self.experiments = [
            Experiment(
                name="XGBoost с подбором параметров",
                algorithm="XGBoost",
                dataset="customer_data.csv",
                project_id=self.projects[0].id
            ),
            Experiment(
                name="Random Forest классификация",
                algorithm="Random Forest",
                dataset="fraud_data.csv",
                project_id=self.projects[1].id
            ),
            Experiment(
                name="LightGBM с GPU",
                algorithm="LightGBM",
                dataset="sales_data.csv",
                project_id=self.projects[2].id
            )
        ]
        
        # Устанавливаем статусы и метрики для демо-экспериментов
        self.experiments[0].status = "completed"
        self.experiments[0].metrics = {"accuracy": 0.92, "precision": 0.89, "recall": 0.91, "f1_score": 0.90}
        
        self.experiments[1].status = "running"
        self.experiments[1].metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1_score": 0.935}
        
        self.experiments[2].status = "created"
        
        # Создаем демо-модели
        self.models = [
            TrainedModel(
                name="Customer Churn Predictor",
                description="Модель для прогнозирования оттока клиентов",
                experiment_id=self.experiments[0].id
            )
        ]
        self.models[0].metrics = self.experiments[0].metrics
        self.models[0].deployment_status = "deployed"
        
        # Сохраняем связи
        for exp in self.experiments:
            for proj in self.projects:
                if exp.project_id == proj.id:
                    proj.experiments.append(exp)
                    break
    
    def get_all_projects(self):
        return self.projects
    
    def get_all_experiments(self):
        return self.experiments
    
    def get_all_models(self):
        return self.models
    
    def get_project_by_id(self, project_id: str):
        for proj in self.projects:
            if proj.id == project_id:
                return proj
        return None
    
    def get_experiment_by_id(self, experiment_id: str):
        for exp in self.experiments:
            if exp.id == experiment_id:
                return exp
        return None
    
    def add_project(self, project: Project):
        self.projects.append(project)
        self._save_to_file()
        return project
    
    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)
        # Находим проект и добавляем в него эксперимент
        project = self.get_project_by_id(experiment.project_id)
        if project:
            project.experiments.append(experiment)
        self._save_to_file()
        return experiment
    
    def update_experiment_status(self, experiment_id: str, status: str, metrics: Dict = None):
        experiment = self.get_experiment_by_id(experiment_id)
        if experiment:
            experiment.status = status
            if status == "running":
                experiment.started_at = datetime.now()
            elif status == "completed":
                experiment.completed_at = datetime.now()
                if metrics:
                    experiment.metrics = metrics
            self._save_to_file()
        return experiment
    
    def _save_to_file(self):
        """Сохраняет данные в JSON файл (для простоты)"""
        data = {
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "status": p.status,
                    "experiment_ids": [e.id for e in p.experiments]
                }
                for p in self.projects
            ],
            "experiments": [
                {
                    "id": e.id,
                    "name": e.name,
                    "algorithm": e.algorithm,
                    "status": e.status,
                    "project_id": e.project_id,
                    "metrics": e.metrics
                }
                for e in self.experiments
            ]
        }
        
        with open("data/database.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

# Инициализируем базу данных
db = Database()

# ============ ВЕБ-ИНТЕРФЕЙС ============

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Главный дашборд"""
    projects = db.get_all_projects()
    experiments = db.get_all_experiments()
    models = db.get_all_models()
    
    # Статистика
    stats = {
        "total_projects": len(projects),
        "total_experiments": len(experiments),
        "total_models": len(models),
        "completed_experiments": len([e for e in experiments if e.status == "completed"]),
        "running_experiments": len([e for e in experiments if e.status == "running"]),
        "deployed_models": len([m for m in models if m.deployment_status == "deployed"])
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "projects": projects,
        "experiments": experiments[:10],  # Последние 10 экспериментов
        "models": models,
        "stats": stats,
        "current_time": datetime.now().strftime("%H:%M")
    })

@app.get("/project/create", response_class=HTMLResponse)
async def create_project_page(request: Request):
    """Страница создания проекта"""
    return templates.TemplateResponse("create_project.html", {
        "request": request
    })

@app.get("/experiment/create", response_class=HTMLResponse)
async def create_experiment_page(request: Request):
    """Страница создания эксперимента"""
    projects = db.get_all_projects()
    return templates.TemplateResponse("create_experiment.html", {
        "request": request,
        "projects": projects
    })

@app.get("/visualization", response_class=HTMLResponse)
async def visualization_page(request: Request):
    """Страница визуализации"""
    experiments = db.get_all_experiments()
    # Готовим данные для графиков
    chart_data = {
        "experiment_names": [e.name[:20] + "..." if len(e.name) > 20 else e.name for e in experiments if e.metrics],
        "accuracy_scores": [e.metrics.get("accuracy", 0) for e in experiments if e.metrics],
        "f1_scores": [e.metrics.get("f1_score", 0) for e in experiments if e.metrics]
    }
    
    return templates.TemplateResponse("visualization.html", {
        "request": request,
        "experiments": experiments,
        "chart_data": json.dumps(chart_data)
    })

@app.get("/project/{project_id}", response_class=HTMLResponse)
async def project_detail(request: Request, project_id: str):
    """Детальная страница проекта"""
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    
    return templates.TemplateResponse("project_detail.html", {
        "request": request,
        "project": project,
        "experiments": [e for e in db.get_all_experiments() if e.project_id == project_id]
    })

@app.get("/experiment/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, experiment_id: str):
    """Детальная страница эксперимента"""
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Эксперимент не найден")
    
    project = db.get_project_by_id(experiment.project_id)
    
    return templates.TemplateResponse("experiment_detail.html", {
        "request": request,
        "experiment": experiment,
        "project": project
    })

# ============ API ENDPOINTS ============

@app.post("/api/projects")
async def create_project_api(
    name: str = Form(...),
    description: str = Form(...),
    tags: str = Form("")
):
    """API для создания проекта"""
    user = db.demo_user  # Используем демо-пользователя
    
    project = Project(
        name=name,
        description=description,
        owner=user
    )
    
    # Добавляем теги
    if tags:
        project.tags = [tag.strip() for tag in tags.split(",")]
    
    db.add_project(project)
    
    return JSONResponse({
        "success": True,
        "message": "Проект успешно создан",
        "project_id": project.id,
        "project_name": project.name
    })

@app.post("/api/experiments")
async def create_experiment_api(
    name: str = Form(...),
    algorithm: str = Form(...),
    dataset: str = Form(...),
    project_id: str = Form(...),
    hyperparameters: str = Form("{}")
):
    """API для создания эксперимента"""
    # Проверяем существование проекта
    project = db.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    
    experiment = Experiment(
        name=name,
        algorithm=algorithm,
        dataset=dataset,
        project_id=project_id
    )
    
    # Парсим гиперпараметры
    try:
        experiment.hyperparameters = json.loads(hyperparameters)
    except:
        experiment.hyperparameters = {}
    
    db.add_experiment(experiment)
    
    return JSONResponse({
        "success": True,
        "message": "Эксперимент успешно создан",
        "experiment_id": experiment.id,
        "experiment_name": experiment.name
    })

@app.post("/api/experiments/{experiment_id}/start")
async def start_experiment_api(experiment_id: str):
    """API для запуска эксперимента"""
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Эксперимент не найден")
    
    # Симулируем обучение с случайными метриками
    import random
    
    # Обновляем статус
    experiment = db.update_experiment_status(experiment_id, "running")
    
    # Генерируем случайные метрики
    metrics = {
        "accuracy": round(random.uniform(0.8, 0.98), 3),
        "precision": round(random.uniform(0.75, 0.96), 3),
        "recall": round(random.uniform(0.78, 0.97), 3),
        "f1_score": round(random.uniform(0.8, 0.96), 3),
        "loss": round(random.uniform(0.1, 0.5), 3),
        "training_time": random.randint(30, 300)  # секунды
    }
    
    # Завершаем эксперимент (в реальности это было бы асинхронно)
    experiment = db.update_experiment_status(experiment_id, "completed", metrics)
    
    return JSONResponse({
        "success": True,
        "message": "Эксперимент завершен успешно",
        "experiment_id": experiment_id,
        "metrics": metrics
    })

@app.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """API для получения метрик эксперимента"""
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Эксперимент не найден")
    
    return JSONResponse({
        "experiment_id": experiment_id,
        "metrics": experiment.metrics
    })

@app.get("/api/stats")
async def get_system_stats():
    """API для получения статистики системы"""
    projects = db.get_all_projects()
    experiments = db.get_all_experiments()
    models = db.get_all_models()
    
    return JSONResponse({
        "projects": len(projects),
        "experiments": len(experiments),
        "models": len(models),
        "completed_experiments": len([e for e in experiments if e.status == "completed"]),
        "running_experiments": len([e for e in experiments if e.status == "running"]),
        "active_projects": len([p for p in projects if p.status == "active"])
    })

# ============ ШАБЛОНЫ HTML ============

# Создаем шаблоны HTML
TEMPLATES = {
    "dashboard.html": """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Platform - Главный дашборд</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        
        header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        h1 { font-size: 32px; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; font-size: 16px; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: transform 0.3s;
        }
        
        .stat-card:hover { transform: translateY(-5px); }
        .stat-card h3 { color: #666; font-size: 14px; text-transform: uppercase; margin-bottom: 10px; }
        .stat-card .value { font-size: 42px; font-weight: bold; color: #2c3e50; }
        
        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        
        .main-content, .widget {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        .sidebar { display: flex; flex-direction: column; gap: 25px; }
        
        h2 { 
            color: #2c3e50; 
            margin-bottom: 20px; 
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            text-align: center;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th { background: #f8f9fa; font-weight: 600; color: #555; }
        
        .status {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-created { background: #f0ad4e; color: white; }
        .status-running { background: #5bc0de; color: white; }
        .status-completed { background: #5cb85c; color: white; }
        
        .metric-badge {
            display: inline-block;
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 10px;
            margin: 2px;
            font-size: 11px;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 ML Platform - Панель управления</h1>
            <p class="subtitle">Система для создания и тестирования алгоритмов машинного обучения</p>
            <p>Текущее время: {{ current_time }}</p>
        </header>
        
        <!-- Статистика -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Проектов</h3>
                <div class="value">{{ stats.total_projects }}</div>
            </div>
            <div class="stat-card">
                <h3>Экспериментов</h3>
                <div class="value">{{ stats.total_experiments }}</div>
            </div>
            <div class="stat-card">
                <h3>Завершено</h3>
                <div class="value">{{ stats.completed_experiments }}</div>
            </div>
            <div class="stat-card">
                <h3>Выполняется</h3>
                <div class="value">{{ stats.running_experiments }}</div>
            </div>
        </div>
        
        <!-- Быстрые действия -->
        <div class="action-buttons">
            <a href="/project/create" class="btn">📁 Создать проект</a>
            <a href="/experiment/create" class="btn">🔬 Создать эксперимент</a>
            <a href="/visualization" class="btn">📊 Визуализация</a>
            <button onclick="refreshDashboard()" class="btn">🔄 Обновить</button>
        </div>
        
        <div class="content-grid">
            <!-- Основной контент -->
            <div class="main-content">
                <h2>📁 Последние проекты</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Название</th>
                            <th>Описание</th>
                            <th>Статус</th>
                            <th>Экспериментов</th>
                            <th>Дата</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for project in projects[:5] %}
                        <tr>
                            <td><a href="/project/{{ project.id }}">{{ project.name }}</a></td>
                            <td>{{ project.description[:50] }}...</td>
                            <td><span class="status status-{{ project.status }}">{{ project.status }}</span></td>
                            <td>{{ project.experiments|length }}</td>
                            <td>{{ project.created_at.strftime('%d.%m.%Y') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                
                <h2 style="margin-top: 30px;">🔬 Последние эксперименты</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Название</th>
                            <th>Алгоритм</th>
                            <th>Статус</th>
                            <th>Метрики</th>
                            <th>Проект</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for exp in experiments %}
                        <tr>
                            <td><a href="/experiment/{{ exp.id }}">{{ exp.name }}</a></td>
                            <td>{{ exp.algorithm }}</td>
                            <td><span class="status status-{{ exp.status }}">{{ exp.status }}</span></td>
                            <td>
                                {% for name, value in exp.metrics.items() %}
                                <span class="metric-badge">{{ name }}: {{ value }}</span>
                                {% endfor %}
                            </td>
                            <td>
                                {% for p in projects %}
                                    {% if p.id == exp.project_id %}
                                        {{ p.name[:20] }}...
                                    {% endif %}
                                {% endfor %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <!-- Боковая панель -->
            <div class="sidebar">
                <div class="widget">
                    <h2>📈 Активность системы</h2>
                    <div class="chart-container">
                        <canvas id="activityChart"></canvas>
                    </div>
                </div>
                
                <div class="widget">
                    <h2>🏆 Лучшие метрики</h2>
                    <div id="best-metrics">
                        {% for exp in experiments %}
                            {% if exp.metrics %}
                                <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                                    <strong>{{ exp.name[:20] }}...</strong><br>
                                    <small>Accuracy: {{ exp.metrics.get('accuracy', 0) }}</small>
                                </div>
                            {% endif %}
                        {% endfor %}
                    </div>
                </div>
                
                <div class="widget">
                    <h2>📋 Обученные модели</h2>
                    {% for model in models %}
                    <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                        <strong>{{ model.name }}</strong><br>
                        <small>Версия: {{ model.version }}</small><br>
                        <small>Статус: {{ model.deployment_status or 'Не развернута' }}</small>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Инициализация графика активности
        const ctx = document.getElementById('activityChart').getContext('2d');
        const activityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
                datasets: [{
                    label: 'Запущено экспериментов',
                    data: [3, 5, 2, 8, 6, 4, 7],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        function refreshDashboard() {
            location.reload();
        }
        
        // Автообновление каждые 30 секунд
        setInterval(refreshDashboard, 30000);
    </script>
</body>
</html>
""",
    
    "create_project.html": """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Создание проекта - ML Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 800px; margin: 50px auto; }
        
        .form-card {
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        h1 { 
            color: #2c3e50; 
            margin-bottom: 30px;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: #667eea;
            outline: none;
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px 28px;
            border-radius: 8px;
            text-decoration: none;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            border: none;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-back {
            background: #6c757d;
            margin-top: 15px;
        }
        
        .message {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-card">
            <h1>📁 Создание нового проекта</h1>
            
            <div id="message" class="message"></div>
            
            <form id="projectForm">
                <div class="form-group">
                    <label for="name">Название проекта *</label>
                    <input type="text" id="name" name="name" required 
                           placeholder="Например: Прогнозирование оттока клиентов">
                </div>
                
                <div class="form-group">
                    <label for="description">Описание проекта *</label>
                    <textarea id="description" name="description" required 
                              placeholder="Опишите цели и задачи проекта..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="tags">Теги (через запятую)</label>
                    <input type="text" id="tags" name="tags" 
                           placeholder="ML, классификация, финансы">
                </div>
                
                <button type="submit" class="btn">Создать проект</button>
                <a href="/" class="btn btn-back">← Назад к дашборду</a>
            </form>
        </div>
    </div>
    
    <script>
        document.getElementById('projectForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const messageDiv = document.getElementById('message');
            
            try {
                const response = await fetch('/api/projects', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    messageDiv.className = 'message success';
                    messageDiv.textContent = `✅ Проект "${result.project_name}" успешно создан!`;
                    messageDiv.style.display = 'block';
                    
                    // Очищаем форму
                    this.reset();
                    
                    // Через 2 секунды переходим к созданию эксперимента
                    setTimeout(() => {
                        window.location.href = `/experiment/create?project_id=${result.project_id}`;
                    }, 2000);
                } else {
                    throw new Error(result.message || 'Ошибка при создании проекта');
                }
            } catch (error) {
                messageDiv.className = 'message error';
                messageDiv.textContent = `❌ Ошибка: ${error.message}`;
                messageDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
""",
    
    "create_experiment.html": """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Создание эксперимента - ML Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 800px; margin: 50px auto; }
        
        .form-card {
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        h1 { 
            color: #2c3e50; 
            margin-bottom: 30px;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: #667eea;
            outline: none;
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px 28px;
            border-radius: 8px;
            text-decoration: none;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            border: none;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-start {
            background: #28a745;
            margin-top: 10px;
        }
        
        .btn-back {
            background: #6c757d;
            margin-top: 15px;
        }
        
        .message {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-card">
            <h1>🔬 Создание нового эксперимента</h1>
            
            <div id="message" class="message"></div>
            
            <form id="experimentForm">
                <div class="form-group">
                    <label for="name">Название эксперимента *</label>
                    <input type="text" id="name" name="name" required 
                           placeholder="Например: XGBoost с подбором гиперпараметров">
                </div>
                
                <div class="form-group">
                    <label for="project_id">Проект *</label>
                    <select id="project_id" name="project_id" required>
                        <option value="">Выберите проект</option>
                        {% for project in projects %}
                        <option value="{{ project.id }}">{{ project.name }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="algorithm">Алгоритм *</label>
                    <select id="algorithm" name="algorithm" required>
                        <option value="">Выберите алгоритм</option>
                        <option value="XGBoost">XGBoost</option>
                        <option value="Random Forest">Random Forest</option>
                        <option value="LightGBM">LightGBM</option>
                        <option value="CatBoost">CatBoost</option>
                        <option value="Logistic Regression">Logistic Regression</option>
                        <option value="Neural Network">Neural Network</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="dataset">Датасет *</label>
                    <select id="dataset" name="dataset" required>
                        <option value="">Выберите датасет</option>
                        <option value="customer_data.csv">Данные клиентов (CSV)</option>
                        <option value="fraud_data.csv">Данные мошенничества (CSV)</option>
                        <option value="sales_data.csv">Данные продаж (CSV)</option>
                        <option value="images_dataset.zip">Набор изображений (ZIP)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="hyperparameters">Гиперпараметры (JSON)</label>
                    <textarea id="hyperparameters" name="hyperparameters" 
                              placeholder='{"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100}'></textarea>
                </div>
                
                <button type="submit" class="btn">Создать эксперимент</button>
                <button type="button" id="startTrainingBtn" class="btn btn-start" style="display: none;">
                    🚀 Запустить обучение
                </button>
                <a href="/" class="btn btn-back">← Назад к дашборду</a>
            </form>
        </div>
    </div>
    
    <script>
        let currentExperimentId = null;
        
        document.getElementById('experimentForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const messageDiv = document.getElementById('message');
            
            try {
                const response = await fetch('/api/experiments', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    messageDiv.className = 'message success';
                    messageDiv.textContent = `✅ Эксперимент "${result.experiment_name}" успешно создан!`;
                    messageDiv.style.display = 'block';
                    
                    currentExperimentId = result.experiment_id;
                    
                    // Показываем кнопку запуска обучения
                    document.getElementById('startTrainingBtn').style.display = 'block';
                    
                    // Прокручиваем к сообщению
                    messageDiv.scrollIntoView({ behavior: 'smooth' });
                } else {
                    throw new Error(result.message || 'Ошибка при создании эксперимента');
                }
            } catch (error) {
                messageDiv.className = 'message error';
                messageDiv.textContent = `❌ Ошибка: ${error.message}`;
                messageDiv.style.display = 'block';
            }
        });
        
        document.getElementById('startTrainingBtn').addEventListener('click', async function() {
            if (!currentExperimentId) return;
            
            const messageDiv = document.getElementById('message');
            
            try {
                const response = await fetch(`/api/experiments/${currentExperimentId}/start`, {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    messageDiv.className = 'message success';
                    messageDiv.innerHTML = `
                        ✅ Обучение завершено успешно!<br>
                        📊 Метрики:<br>
                        ${Object.entries(result.metrics).map(([k, v]) => 
                            `• ${k}: ${v}<br>`
                        ).join('')}
                        <a href="/experiment/${currentExperimentId}" class="btn" style="margin-top: 10px;">
                            📄 Перейти к деталям эксперимента
                        </a>
                    `;
                    messageDiv.style.display = 'block';
                    
                    // Скрываем кнопку запуска
                    this.style.display = 'none';
                } else {
                    throw new Error(result.message || 'Ошибка при запуске обучения');
                }
            } catch (error) {
                messageDiv.className = 'message error';
                messageDiv.textContent = `❌ Ошибка: ${error.message}`;
                messageDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
""",
    
    "visualization.html": """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Визуализация - ML Platform</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        
        header { 
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        h1 { font-size: 32px; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; font-size: 16px; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .chart-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
        
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            text-align: center;
            font-weight: 600;
            transition: all 0.3s;
            margin: 10px 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        h2 { 
            color: #2c3e50; 
            margin-bottom: 20px; 
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th { background: #f8f9fa; font-weight: 600; color: #555; }
        
        .metric-badge {
            display: inline-block;
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 10px;
            margin: 2px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Визуализация результатов</h1>
            <p class="subtitle">Анализ и сравнение метрик экспериментов</p>
            <div>
                <a href="/" class="btn">← Назад к дашборду</a>
                <button onclick="refreshCharts()" class="btn">🔄 Обновить графики</button>
            </div>
        </header>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h2>📈 Сравнение точности (Accuracy)</h2>
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h2>🎯 Сравнение F1-Score</h2>
                <div class="chart-container">
                    <canvas id="f1Chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h2>📉 Матрица ошибок (Confusion Matrix)</h2>
                <div class="chart-container">
                    <canvas id="confusionMatrix"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h2>📊 Распределение метрик</h2>
                <div class="chart-container">
                    <canvas id="metricsDistribution"></canvas>
                </div>
            </div>
        </div>
        
        <div class="chart-card" style="margin-top: 25px;">
            <h2>📋 Детальные метрики экспериментов</h2>
            <table>
                <thead>
                    <tr>
                        <th>Эксперимент</th>
                        <th>Алгоритм</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Статус</th>
                    </tr>
                </thead>
                <tbody>
                    {% for exp in experiments %}
                    <tr>
                        <td><a href="/experiment/{{ exp.id }}">{{ exp.name }}</a></td>
                        <td>{{ exp.algorithm }}</td>
                        <td>{{ exp.metrics.get('accuracy', 'N/A') }}</td>
                        <td>{{ exp.metrics.get('precision', 'N/A') }}</td>
                        <td>{{ exp.metrics.get('recall', 'N/A') }}</td>
                        <td>{{ exp.metrics.get('f1_score', 'N/A') }}</td>
                        <td>{{ exp.status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        const chartData = {{ chart_data|safe }};
        
        // График точности
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        const accuracyChart = new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: chartData.experiment_names,
                datasets: [{
                    label: 'Accuracy',
                    data: chartData.accuracy_scores,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // График F1-Score
        const f1Ctx = document.getElementById('f1Chart').getContext('2d');
        const f1Chart = new Chart(f1Ctx, {
            type: 'line',
            data: {
                labels: chartData.experiment_names,
                datasets: [{
                    label: 'F1-Score',
                    data: chartData.f1_scores,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
        
        // Матрица ошибок
        const matrixCtx = document.getElementById('confusionMatrix').getContext('2d');
        const confusionMatrix = new Chart(matrixCtx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Confusion Matrix',
                    data: [
                        {x: 'True Positive', y: 'Predicted Positive', v: 85},
                        {x: 'False Negative', y: 'Predicted Positive', v: 15},
                        {x: 'False Positive', y: 'Predicted Negative', v: 10},
                        {x: 'True Negative', y: 'Predicted Negative', v: 90}
                    ],
                    backgroundColor: function(context) {
                        const value = context.dataset.data[context.dataIndex].v;
                        const alpha = value / 100;
                        return `rgba(255, 99, 132, ${alpha})`;
                    },
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                    width: 100,
                    height: 100
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.x}, ${context.raw.y}: ${context.raw.v}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'category',
                        labels: ['Predicted Positive', 'Predicted Negative'],
                        offset: true,
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        type: 'category',
                        labels: ['True Positive', 'False Negative', 'False Positive', 'True Negative'],
                        offset: true,
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
        
        // Распределение метрик
        const distCtx = document.getElementById('metricsDistribution').getContext('2d');
        const metricsDistribution = new Chart(distCtx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'Memory'],
                datasets: [
                    {
                        label: 'Лучший эксперимент',
                        data: [0.95, 0.92, 0.93, 0.94, 0.85, 0.78],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                    },
                    {
                        label: 'Средние значения',
                        data: [0.85, 0.82, 0.83, 0.84, 0.75, 0.65],
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        pointBackgroundColor: 'rgba(54, 162, 235, 1)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
        
        function refreshCharts() {
            accuracyChart.update();
            f1Chart.update();
            confusionMatrix.update();
            metricsDistribution.update();
        }
    </script>
</body>
</html>
"""
}

# Сохраняем шаблоны
for filename, content in TEMPLATES.items():
    with open(f"templates/{filename}", "w", encoding="utf-8") as f:
        f.write(content)

# Создаем простой статический файл для стилей
os.makedirs("static/css", exist_ok=True)
with open("static/css/style.css", "w", encoding="utf-8") as f:
    f.write("""
    .experiment-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: white;
    }
    
    .metric {
        display: inline-block;
        background: #f0f0f0;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 12px;
    }
    """)

# ============ ЗАПУСК ПРИЛОЖЕНИЯ ============

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 ML Platform - Веб-интерфейс запускается...")
    print("📡 Откройте браузер и перейдите по адресу:")
    print("   http://localhost:8000")
    print("="*60)
    
    # Используем строку для импорта вместо объекта
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)