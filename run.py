"""
Простой запуск ML платформы
"""

import os
import sys

# Получаем текущую директорию
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к ml_platform
ml_platform_path = os.path.join(current_dir, "ml_platform")

print("🚀 Запуск ML платформы...")
print(f"📁 Корневая директория: {current_dir}")
print(f"📁 Путь к ml_platform: {ml_platform_path}")

# Меняем рабочую директорию на ml_platform
os.chdir(ml_platform_path)

try:
    # Запускаем main.py с указанием кодировки UTF-8
    with open("main.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code)
except UnicodeDecodeError:
    # Если не получается с UTF-8, пробуем другие кодировки
    print("⚠️ Проблема с кодировкой UTF-8, пробуем cp1251...")
    with open("main.py", "r", encoding="cp1251") as f:
        code = f.read()
    exec(code)
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("🔄 Пробуем другой способ...")
    
    # Запускаем через subprocess
    import subprocess
    subprocess.run([sys.executable, "main.py"])