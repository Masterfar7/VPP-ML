# VPP-ML: DDoS & PortScan Detection System
Система машинного обучения для обнаружения сетевых атак DDoS и PortScan на основе анализа сетевого трафика.

## Описание проекта

Этот проект представляет собой систему детектирования сетевых атак, которая использует алгоритмы машинного обучения для классификации сетевого трафика на:
DDoS, PortScan атаки, нормальный трафик.

## Реализация

Спроектировал train.py в коотором использовал классификатор Random Forest, чтобы на основе множества решений обучить модель, обучена модель на датасетах: final_dataset_small.csv, balanced_50_50_small.csv(DDoS) и scan_small.json, они урезаны, чтобы можно было быстро загрузить на github. Infering.py загружает обученную модель, анализирует переданный CSV-файл с сетевым трафиком и выдает детальный отчет о том, сколько атак обнаружено с метриками точности: Accuracy, Precision, Recall, F1-Score, Total Records, Actual Attacks, Predicted Attacks, DDoS Count, PortScan Count, Normal Count. Detect.cpp принимает dataset, который нужно проанализировать и вызывает Infering.py и отображает полученные результаты.

Датасеты взял отсюда:
https://www.kaggle.com/datasets/devendra416/ddos-datasets?resource=download
https://www.kaggle.com/datasets/signalspikes/internet-port-scan-1

## Как запустить проект

1. Клонировать репозиторий
2. Перейти в папку VPP-ML
3. Открыть консоль в этой папке
4. Выполнить chmod +x setup.sh
5. Выполнить ./setup.sh (Установка необходимых зависимостей и библиотек)
6. Выполнить source ml_env/bin/activate (активирование окружающей среды python)
7. python3 train.py (Обучение модели)
8. g++ -o detect detect.cpp (Компиляция исполняемого файла)
9. ./detect path_to_dataset (вместо path_to_dataset нужно вставить путь к файлу для анализа, пример ./detect datasets/balanced_50_50_small.csv)
Также сделал тестовый файл mixed_30_30_40_test.csv для проверки работоспособности модели.

