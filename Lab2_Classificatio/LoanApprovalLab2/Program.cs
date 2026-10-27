using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

Console.OutputEncoding = Encoding.UTF8;

try
{
    string csvPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "loan.csv");
    if (!File.Exists(csvPath))
    {
        Console.WriteLine("Не знайдено data/loan.csv.");
        Console.WriteLine("Завантажте навчальний CSV із Kaggle, який містить Loan_Status,");
        Console.WriteLine("покладіть його в папку data та перейменуйте на loan.csv.");
        return;
    }

    var records = ReadLoans(csvPath);
    Console.WriteLine($"Прочитано розмічених записів: {records.Count}");
    Console.WriteLine($"Схвалено (Y): {records.Count(r => r.Label)}");
    Console.WriteLine($"Відмовлено (N): {records.Count(r => !r.Label)}");

    if (records.Count < 20 || records.All(r => r.Label) || records.All(r => !r.Label))
        throw new InvalidOperationException("Для навчання потрібно щонайменше 20 записів і обидва класи Y/N.");

    // Поділяємо дані ДО заповнення пропусків, щоб тестова вибірка не впливала на навчання.
    var random = new Random(42);
    var shuffled = records.OrderBy(_ => random.Next()).ToList();
    int trainCount = (int)(shuffled.Count * 0.8);
    var rawTrain = shuffled.Take(trainCount).ToList();
    var rawTest = shuffled.Skip(trainCount).ToList();
    var fill = ImputationValues.FromTrainingRows(rawTrain);

    var ml = new MLContext(seed: 42);
    IDataView trainData = ml.Data.LoadFromEnumerable(rawTrain.Select(row => ToLoanData(row, fill)));
    IDataView testData = ml.Data.LoadFromEnumerable(rawTest.Select(row => ToLoanData(row, fill)));

    Console.WriteLine($"Навчальна вибірка: {rawTrain.Count}; тестова: {rawTest.Count}");
    Console.WriteLine("Заповнення пропусків: медіана числових полів, мода кредитної історії, Unknown для категорій.");

    // Перетворюємо текстові ознаки в числа, об'єднуємо й нормалізуємо їх.
    var pipeline = ml.Transforms.Categorical.OneHotEncoding("GenderEncoded", nameof(LoanData.Gender))
        .Append(ml.Transforms.Categorical.OneHotEncoding("MarriedEncoded", nameof(LoanData.Married)))
        .Append(ml.Transforms.Categorical.OneHotEncoding("DependentsEncoded", nameof(LoanData.Dependents)))
        .Append(ml.Transforms.Categorical.OneHotEncoding("EducationEncoded", nameof(LoanData.Education)))
        .Append(ml.Transforms.Categorical.OneHotEncoding("SelfEmployedEncoded", nameof(LoanData.SelfEmployed)))
        .Append(ml.Transforms.Categorical.OneHotEncoding("PropertyAreaEncoded", nameof(LoanData.PropertyArea)))
        .Append(ml.Transforms.Concatenate("Features",
            "GenderEncoded", "MarriedEncoded", "DependentsEncoded", "EducationEncoded",
            "SelfEmployedEncoded", "PropertyAreaEncoded",
            nameof(LoanData.ApplicantIncome), nameof(LoanData.CoapplicantIncome),
            nameof(LoanData.LoanAmount), nameof(LoanData.LoanAmountTerm), nameof(LoanData.CreditHistory)))
        .Append(ml.Transforms.NormalizeMinMax("Features"))
        .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(LoanData.Label), featureColumnName: "Features"));

    Console.WriteLine("Навчання моделі...");
    ITransformer model = pipeline.Fit(trainData);
    var predictions = model.Transform(testData);
    var metrics = ml.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(LoanData.Label));

    var report = new StringBuilder();
    report.AppendLine("Лабораторна робота №2 — Прогнозування схвалення кредиту");
    report.AppendLine($"Дата запуску: {DateTime.Now:yyyy-MM-dd HH:mm}");
    report.AppendLine($"Усього записів: {records.Count}");
    report.AppendLine($"Навчальна вибірка: {rawTrain.Count}");
    report.AppendLine($"Тестова вибірка: {rawTest.Count}");
    report.AppendLine("Алгоритм: SdcaLogisticRegressionBinaryTrainer");
    report.AppendLine($"Accuracy:  {metrics.Accuracy:P2}");
    report.AppendLine($"Precision: {metrics.PositivePrecision:P2}");
    report.AppendLine($"Recall:    {metrics.PositiveRecall:P2}");
    report.AppendLine($"F1Score:   {metrics.F1Score:P2}");
    report.AppendLine($"ROC AUC:   {metrics.AreaUnderRocCurve:F4}");

    Console.WriteLine();
    Console.WriteLine(report);
    File.WriteAllText("metrics.txt", report.ToString(), Encoding.UTF8);

    // Збереження навченої моделі (прогнозний конвеєр включений).
    ml.Model.Save(model, trainData.Schema, "model.zip");
    Console.WriteLine("Модель збережено в model.zip; метрики — у metrics.txt.");

    // Демонстраційний прогноз для умовного позичальника.
    var example = new LoanData
    {
        Gender = "Male", Married = "Yes", Dependents = "0", Education = "Graduate",
        SelfEmployed = "No", ApplicantIncome = 5500, CoapplicantIncome = 1000,
        LoanAmount = 125, LoanAmountTerm = 360, CreditHistory = 1,
        PropertyArea = "Urban"
    };
    var engine = ml.Model.CreatePredictionEngine<LoanData, LoanPrediction>(model);
    LoanPrediction prediction = engine.Predict(example);

    Console.WriteLine("\nПриклад прогнозу для умовного клієнта:");
    Console.WriteLine($"Прогноз: {(prediction.PredictedLabel ? "схвалено (Y)" : "відхилено (N)")}");
    Console.WriteLine($"Оціночна ймовірність класу Y: {prediction.Probability:P2}");
    Console.WriteLine("Увага: це навчальна модель, а не підстава для реального кредитного рішення.");
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Помилка: {ex.Message}");
    Environment.ExitCode = 1;
}

static List<RawLoan> ReadLoans(string path)
{
    // Спрощений CSV-читач саме для Kaggle Loan Data Set: значення не містять ком.
    using var reader = new StreamReader(path);
    string? headerLine = reader.ReadLine();
    if (headerLine is null) throw new InvalidOperationException("CSV-файл порожній.");

    var headers = headerLine.TrimStart('\uFEFF').Split(',')
        .Select(h => h.Trim().Trim('"')).ToArray();
    var index = headers.Select((name, position) => (name, position))
        .ToDictionary(x => x.name, x => x.position, StringComparer.OrdinalIgnoreCase);

    string[] required = { "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "Loan_Status" };
    foreach (var field in required)
        if (!index.ContainsKey(field))
            throw new InvalidOperationException($"У CSV немає колонки {field}. Потрібен навчальний файл з Loan_Status.");

    var list = new List<RawLoan>();
    string? line;
    int lineNumber = 1;
    while ((line = reader.ReadLine()) is not null)
    {
        lineNumber++;
        if (string.IsNullOrWhiteSpace(line)) continue;
        var values = line.Split(',');
        if (values.Length != headers.Length)
            throw new InvalidOperationException($"Некоректний CSV-рядок {lineNumber}: інша кількість колонок.");
        string Get(string name) => values[index[name]].Trim().Trim('"').Trim();

        string label = Get("Loan_Status").ToUpperInvariant();
        if (label is not ("Y" or "N")) continue; // пропускаємо записи без правильної мітки

        list.Add(new RawLoan(
            Clean(Get("Gender")), Clean(Get("Married")), Clean(Get("Dependents")),
            Clean(Get("Education")), Clean(Get("Self_Employed")),
            Number(Get("ApplicantIncome")), Number(Get("CoapplicantIncome")),
            Number(Get("LoanAmount")), Number(Get("Loan_Amount_Term")),
            Number(Get("Credit_History")), Clean(Get("Property_Area")), label == "Y"));
    }
    return list;
}

static string Clean(string input) =>
    string.IsNullOrWhiteSpace(input) || input.Equals("NA", StringComparison.OrdinalIgnoreCase)
        ? "Unknown" : input;

static float? Number(string input) =>
    float.TryParse(input, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
    && float.IsFinite(value) ? value : null;

static LoanData ToLoanData(RawLoan r, ImputationValues fill) => new()
{
    Gender = r.Gender,
    Married = r.Married,
    Dependents = r.Dependents,
    Education = r.Education,
    SelfEmployed = r.SelfEmployed,
    PropertyArea = r.PropertyArea,
    ApplicantIncome = r.ApplicantIncome ?? fill.ApplicantIncome,
    CoapplicantIncome = r.CoapplicantIncome ?? fill.CoapplicantIncome,
    LoanAmount = r.LoanAmount ?? fill.LoanAmount,
    LoanAmountTerm = r.LoanAmountTerm ?? fill.LoanAmountTerm,
    CreditHistory = r.CreditHistory ?? fill.CreditHistory,
    Label = r.Label
};

internal sealed record RawLoan(
    string Gender, string Married, string Dependents, string Education,
    string SelfEmployed, float? ApplicantIncome, float? CoapplicantIncome,
    float? LoanAmount, float? LoanAmountTerm, float? CreditHistory,
    string PropertyArea, bool Label);

internal sealed class ImputationValues
{
    public float ApplicantIncome { get; init; }
    public float CoapplicantIncome { get; init; }
    public float LoanAmount { get; init; }
    public float LoanAmountTerm { get; init; }
    public float CreditHistory { get; init; }

    public static ImputationValues FromTrainingRows(List<RawLoan> rows) => new()
    {
        ApplicantIncome = Median(rows.Select(r => r.ApplicantIncome)),
        CoapplicantIncome = Median(rows.Select(r => r.CoapplicantIncome)),
        LoanAmount = Median(rows.Select(r => r.LoanAmount)),
        LoanAmountTerm = Median(rows.Select(r => r.LoanAmountTerm)),
        CreditHistory = rows.Where(r => r.CreditHistory.HasValue)
            .GroupBy(r => r.CreditHistory!.Value)
            .OrderByDescending(g => g.Count())
            .Select(g => g.Key).FirstOrDefault()
    };

    private static float Median(IEnumerable<float?> source)
    {
        var ordered = source.Where(x => x.HasValue).Select(x => x!.Value).Order().ToArray();
        if (ordered.Length == 0) return 0;
        int n = ordered.Length;
        return n % 2 == 1 ? ordered[n / 2] : (ordered[n / 2 - 1] + ordered[n / 2]) / 2;
    }
}

public sealed class LoanData
{
    public string Gender { get; set; } = "Unknown";
    public string Married { get; set; } = "Unknown";
    public string Dependents { get; set; } = "Unknown";
    public string Education { get; set; } = "Unknown";
    public string SelfEmployed { get; set; } = "Unknown";
    public string PropertyArea { get; set; } = "Unknown";
    public float ApplicantIncome { get; set; }
    public float CoapplicantIncome { get; set; }
    public float LoanAmount { get; set; }
    public float LoanAmountTerm { get; set; }
    public float CreditHistory { get; set; }
    public bool Label { get; set; }
}

public sealed class LoanPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
