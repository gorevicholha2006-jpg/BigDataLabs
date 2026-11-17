# Лабораторна робота №3 — Визначення аномалій

**Дисципліна:** Обробка структурованих та неструктурованих даних (BigData)  
**Варіант 1:** Контрольні показники — аномалії сезонних KPI  
**Основна технологія:** C# / .NET 8 / ML.NET 4.0.1 / `RandomizedPcaTrainer`

## Коротко про проєкт

Модель аналізує синтетичний часовий ряд щоденних замовлень інтернет-магазину (KPI) і визначає нетипові відхилення з урахуванням тижневої сезонності. В умові лабораторної дозволено **власний набір даних** замість Kaggle. Тому у проєкті міститься відтворюваний CSV-файл на 300 дат (2025-01-01 — 2025-10-27) з 12 навмисно згенерованими аномаліями.

Тренувальний/тестовий поділ **хронологічний**: перші 240 дат — тренувальна частина, останні 60 — тестова. Для обчислення сезонних ознак перші 56 днів використовуються тільки як історія. З тренувальної частини виключаються **позначені аномалії**; на тестовій перевіряються всі точки.

## Структура

```text
Lab3_AnomalyDetection/
├── Lab3_AnomalyDetection.csproj  # проєкт .NET 8, Microsoft.ML 4.0.1
├── Program.cs                   # навчання, оцінювання, збереження моделі
├── data/
│   ├── kpi_seasonal_synthetic.csv
│   ├── generate_dataset.py      # відтворює датасет, лише стандартна бібліотека Python
│   ├── reference_check.py       # необов'язкова перевірка PCA на scikit-learn
│   └── plot_model_results.py    # графік реальних прогнозів після dotnet run
├── figures/
│   ├── kpi_timeseries.png
│   └── weekday_seasonality.png
├── results/
│   └── .gitkeep                  # після запуску тут з'являться результати ML.NET
└── report/
    └── Lab3_AnomalyDetection_Report.docx
```

## Як запустити на MacBook

1. Установіть **.NET SDK 8** для вашої архітектури Mac з [офіційного сайту .NET](https://dotnet.microsoft.com/download/dotnet/8.0).
2. Відкрийте Terminal в папці `Lab3_AnomalyDetection` і перевірте `dotnet --version`.
3. Виконайте:

```bash
dotnet restore
dotnet run
```

За першого запуску `dotnet restore` потрібне з'єднання з інтернетом для NuGet. Команда `dotnet run`:

- навчає `RandomizedPcaTrainer` на **183 нормальних прикладах** (після початкового історичного вікна);
- прогнозує аномалії на **60 тестових днях**, з яких 10 мають мітку аномалії;
- друкує **TP, FP, TN, FN, Accuracy, Precision, Recall, F1**;
- показує 10 днів з найбільшим `Score`;
- створює `results/model.zip`, `results/predictions.csv`, `results/metrics.txt`.

**Важливо:** `results/model.zip` — це файл навченої моделі ML.NET, який з'являється **після фактичного запуску** коду. Він не є частиною початкового архіву.

## Які ознаки отримує модель

Для кожного дня, починаючи з 57-го, обчислюється вектор із чотирьох числових показників:

1. `KPI(t)` — замовлення поточного дня;
2. `WeekdayMedian(t)` — медіана восьми **попередніх** однойменних днів тижня;
3. `RollingMedian28(t)` — медіана **попередніх** 28 днів;
4. `KPI(t-7)` — значення цього самого дня тижня попереднього тижня.

Усі ознаки будуються **без майбутніх значень**; `IsAnomaly` не передається моделі як ознака. Вектор нормалізується `NormalizeMeanVariance`, а потім тренується PCA (`rank=3`, `oversampling=1`, `seed=42`). За замовчуванням ML.NET позначає `PredictedLabel=true`, якщо `Score > 0.5`.

Для побудови графіка **фактичних** прогнозів ML.NET після `dotnet run` (потрібні `pandas` та `matplotlib`):

```bash
python3 data/plot_model_results.py
```

## Контрольна перевірка (не плутати з ML.NET)

`data/reference_check.py` незалежно перевіряє сформовані ознаки на **класичному PCA з бібліотеки scikit-learn**. Його результат на 60 тестових прикладах: **TP=10, TN=48, FP=2, FN=0, Precision=83.33%, Recall=100%, F1=90.91%**. Ці цифри **не є результатом виконання ML.NET**; вони підтверджують, що набір даних і ознаки придатні для PCA-підходу. Остаточні метрики `RandomizedPcaTrainer` потрібно брати тільки з `results/metrics.txt` після запуску `dotnet run`.

За бажанням перевірку можна повторити після встановлення Python-бібліотек `numpy`, `pandas`, `scikit-learn`:

```bash
python3 data/reference_check.py
```

## Як розповісти про роботу на захисті

Мета — знайти аномальні KPI, які відрізняються від звичайної тижневої сезонності. Тому порівнюємо поточне значення не лише із загальним середнім, а з медіаною попередніх аналогічних днів тижня. Обираємо `RandomizedPcaTrainer`, як вказано в лекції. Ділимо дані за датою, щоб модель не бачила майбутнього під час навчання. Тренуємо на нормальних записах, а потім перевіряємо, чи знаходить вона позначені аномалії в останніх 60 днях. Метрики Precision і Recall показують відповідно частку правильних тривог та частку знайдених реальних аномалій.

## Джерела

- Методичні вказівки до лабораторної №3 (знімки сторінок 37–38, надані користувачем).
- [Microsoft: RandomizedPcaTrainer](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.randomizedpcatrainer?view=ml-dotnet)
- [Microsoft: RandomizedPca API](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.pcacatalog.randomizedpca?view=ml-dotnet)
- [Офіційний приклад ML.NET anomaly detection](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/AnomalyDetection_CreditCardFraudDetection)
- Датасет, запропонований у завданні (альтернатива, **не використаний**): [Kaggle — benchmark-labeled-anomaly-detection-ts](https://www.kaggle.com/datasets/caesarlupum/benchmark-labeled-anomaly-detection-ts).

## Статус перевірки

- ✅ CSV створений, перевірена кількість рядків, міток і хронологічний поділ.
- ✅ Сезонні ознаки й незалежна PCA-перевірка виконані на Python.
- ✅ Звіт і графіки сформовані.
- ⚠️ У середовищі підготовки проєкту **не встановлено .NET SDK**, тому фактичне збирання/запуск C# і результат саме ML.NET тут не перевірялися. Не слід видавати контрольні Python-метрики за результати ML.NET.
