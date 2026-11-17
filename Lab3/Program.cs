using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

// Lab 3: anomaly detection in seasonal daily business KPIs.
// The training is unsupervised: only known-normal points are fed to RandomizedPca.
// Ground-truth labels are used ONLY for filtering normal training points and test metrics.

try
{
    var projectRoot = FindProjectRoot();
    var csvPath = Path.Combine(projectRoot, "data", "kpi_seasonal_synthetic.csv");
    var resultsDirectory = Path.Combine(projectRoot, "results");
    Directory.CreateDirectory(resultsDirectory);

    List<KpiRecord> raw = ReadKpiCsv(csvPath);
    if (raw.Count < 100)
        throw new InvalidOperationException("At least 100 daily observations are required.");

    // All rolling and seasonal features depend on the past only (no future look-ahead).
    const int warmupDays = 56; // 8 preceding observations for the same weekday
    const int trainEndExclusive = 240; // chronological 80/20 split of raw daily records
    if (raw.Count != 300)
        throw new InvalidOperationException("The provided example expects 300 rows.");

    List<PreparedPoint> examples = PrepareFeatures(raw, warmupDays);
    var trainRows = examples.Where(p => p.DayIndex < trainEndExclusive).ToList();
    var trainNormalRows = trainRows.Where(p => !p.ActualIsAnomaly).ToList();
    var testRows = examples.Where(p => p.DayIndex >= trainEndExclusive).ToList();

    if (trainNormalRows.Count < 50 || testRows.Count == 0)
        throw new InvalidOperationException("Insufficient training/test data after preprocessing.");

    MLContext mlContext = new(seed: 42);
    var trainView = mlContext.Data.LoadFromEnumerable(
        trainNormalRows.Select(p => new ModelInput { Features = p.Features }));
    var testView = mlContext.Data.LoadFromEnumerable(
        testRows.Select(p => new ModelInput { Features = p.Features }));

    // Normalize on training data only; the learned means/scales are reused for test data.
    var pipeline = mlContext.Transforms
        .NormalizeMeanVariance(outputColumnName: "NormalizedFeatures", inputColumnName: "Features")
        .Append(mlContext.AnomalyDetection.Trainers.RandomizedPca(
            featureColumnName: "NormalizedFeatures",
            rank: 3,
            oversampling: 1,
            ensureZeroMean: true,
            seed: 42));

    ITransformer model = pipeline.Fit(trainView);
    var scoredView = model.Transform(testView);
    var predictions = mlContext.Data
        .CreateEnumerable<AnomalyPrediction>(scoredView, reuseRowObject: false)
        .ToList();

    if (predictions.Count != testRows.Count)
        throw new InvalidOperationException("Prediction count differs from test count.");

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < testRows.Count; i++)
    {
        bool actual = testRows[i].ActualIsAnomaly;
        bool predicted = predictions[i].PredictedLabel; // default PCA threshold: 0.5
        if (actual && predicted) tp++;
        else if (!actual && !predicted) tn++;
        else if (!actual && predicted) fp++;
        else fn++;
    }

    double precision = DivideOrZero(tp, tp + fp);
    double recall = DivideOrZero(tp, tp + fn);
    double f1 = DivideOrZero(2 * precision * recall, precision + recall);
    double accuracy = DivideOrZero(tp + tn, testRows.Count);

    string modelPath = Path.Combine(resultsDirectory, "model.zip");
    mlContext.Model.Save(model, trainView.Schema, modelPath);

    string predictionCsv = Path.Combine(resultsDirectory, "predictions.csv");
    using (var writer = new StreamWriter(predictionCsv, false, Encoding.UTF8))
    {
        writer.WriteLine("Date,KpiValue,ActualIsAnomaly,PredictedIsAnomaly,Score");
        for (int i = 0; i < testRows.Count; i++)
        {
            var row = testRows[i];
            var prediction = predictions[i];
            writer.WriteLine(string.Join(",",
                row.Date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                row.KpiValue.ToString(CultureInfo.InvariantCulture),
                row.ActualIsAnomaly ? "1" : "0",
                prediction.PredictedLabel ? "1" : "0",
                prediction.Score.ToString("G9", CultureInfo.InvariantCulture)));
        }
    }

    var metrics = new StringBuilder();
    metrics.AppendLine("LAB 3 — Seasonal KPI anomaly detection, ML.NET RandomizedPca");
    metrics.AppendLine($"Raw observations: {raw.Count}");
    metrics.AppendLine($"First available features: day index {warmupDays} (past-only lookback)");
    metrics.AppendLine($"Chronological train: days 0..{trainEndExclusive - 1}");
    metrics.AppendLine($"Train rows after warm-up: {trainRows.Count}");
    metrics.AppendLine($"Known-normal training rows: {trainNormalRows.Count}");
    metrics.AppendLine($"Excluded labeled training anomalies: {trainRows.Count - trainNormalRows.Count}");
    metrics.AppendLine($"Test rows: {testRows.Count}");
    metrics.AppendLine($"Known test anomalies: {testRows.Count(p => p.ActualIsAnomaly)}");
    metrics.AppendLine("Model: RandomizedPca(rank=3, oversampling=1, seed=42)");
    metrics.AppendLine("Normalization: mean/variance fitted on normal training data");
    metrics.AppendLine("Prediction threshold: built-in default 0.5 (not tuned on test data)");
    metrics.AppendLine($"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}");
    metrics.AppendLine($"Accuracy:  {accuracy:P2}");
    metrics.AppendLine($"Precision: {precision:P2}");
    metrics.AppendLine($"Recall:    {recall:P2}");
    metrics.AppendLine($"F1-score:  {f1:P2}");

    Console.OutputEncoding = Encoding.UTF8;
    Console.WriteLine(metrics.ToString());
    File.WriteAllText(Path.Combine(resultsDirectory, "metrics.txt"), metrics.ToString(), Encoding.UTF8);
    Console.WriteLine("Top 10 highest anomaly scores on test data:");
    foreach (var top in testRows
        .Select((p, i) => new { Point = p, Prediction = predictions[i] })
        .OrderByDescending(x => x.Prediction.Score)
        .Take(10))
    {
        Console.WriteLine($"  {top.Point.Date:yyyy-MM-dd}: " +
                          $"KPI={top.Point.KpiValue,4}; " +
                          $"score={top.Prediction.Score:F4}; " +
                          $"predicted={(top.Prediction.PredictedLabel ? 1 : 0)}; " +
                          $"actual={(top.Point.ActualIsAnomaly ? 1 : 0)}");
    }

    Console.WriteLine($"\nSaved model: {modelPath}");
    Console.WriteLine($"Saved test predictions: {predictionCsv}");
    Console.WriteLine($"Saved metrics: {Path.Combine(resultsDirectory, "metrics.txt")}");
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    Environment.ExitCode = 1;
}

static string FindProjectRoot()
{
    var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
    while (directory is not null)
    {
        if (File.Exists(Path.Combine(directory.FullName, "Lab3_AnomalyDetection.csproj")))
            return directory.FullName;
        directory = directory.Parent;
    }
    throw new DirectoryNotFoundException("Start the app from the Lab3_AnomalyDetection project folder.");
}

static List<KpiRecord> ReadKpiCsv(string path)
{
    if (!File.Exists(path))
        throw new FileNotFoundException("KPI dataset not found.", path);

    string[] lines = File.ReadAllLines(path, Encoding.UTF8);
    var records = new List<KpiRecord>();
    for (int i = 1; i < lines.Length; i++)
    {
        if (string.IsNullOrWhiteSpace(lines[i])) continue;
        string[] parts = lines[i].Split(',');
        if (parts.Length != 4)
            throw new FormatException($"Malformed CSV at line {i + 1}.");
        records.Add(new KpiRecord(
            DateTime.ParseExact(parts[0], "yyyy-MM-dd", CultureInfo.InvariantCulture),
            float.Parse(parts[1], CultureInfo.InvariantCulture),
            parts[2] == "1"));
    }
    records.Sort((a, b) => a.Date.CompareTo(b.Date));
    for (int i = 1; i < records.Count; i++)
        if ((records[i].Date - records[i - 1].Date).TotalDays != 1)
            throw new FormatException("The daily date series must be continuous and unique.");
    return records;
}

static List<PreparedPoint> PrepareFeatures(List<KpiRecord> records, int lookback)
{
    var output = new List<PreparedPoint>();
    for (int i = lookback; i < records.Count; i++)
    {
        // The 8 preceding values of the SAME weekday model the weekly cycle.
        var weeklyHistory = Enumerable.Range(1, 8)
            .Select(week => records[i - 7 * week].Value).ToList();
        float weekdayMedian = Median(weeklyHistory);

        // Historical 28-day median gives a robust local level.
        var recentHistory = records.Skip(i - 28).Take(28).Select(x => x.Value).ToList();
        float recentMedian = Median(recentHistory);
        float today = records[i].Value;
        float previousWeek = records[i - 7].Value;

        // Normal seasonal KPIs have a stable relationship across these four
        // indicators. A sudden change in today's value breaks that relationship.
        // No label or future observation enters the features.
        float[] features =
        [
            today,
            weekdayMedian,
            recentMedian,
            previousWeek
        ];
        output.Add(new PreparedPoint(i, records[i].Date, today, records[i].IsAnomaly, features));
    }
    return output;
}

static float Median(List<float> values)
{
    if (values.Count == 0) throw new ArgumentException("Cannot calculate median of an empty list.");
    values.Sort();
    int mid = values.Count / 2;
    return values.Count % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2f;
}

static double DivideOrZero(double numerator, double denominator) =>
    denominator <= 0 ? 0 : numerator / denominator;

public sealed record KpiRecord(DateTime Date, float Value, bool IsAnomaly);
public sealed record PreparedPoint(int DayIndex, DateTime Date, float KpiValue,
                                   bool ActualIsAnomaly, float[] Features);

public sealed class ModelInput
{
    [VectorType(4)]
    public float[] Features { get; set; } = new float[4];
}

public sealed class AnomalyPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }
}
