using Microsoft.ML;
using Microsoft.ML.Data;

string dataPath = "TSLA.csv";
string modelPath = "model.zip";

MLContext mlContext = new MLContext(seed: 0);

while (true)
{
    Console.WriteLine();
    Console.WriteLine("=== Меню ===");
    Console.WriteLine("1. Натренувати нову модель");
    Console.WriteLine("2. Завантажити вже натреновану модель");
    Console.WriteLine("0. Вийти");
    Console.Write("Оберіть пункт: ");

    string choice = Console.ReadLine();

    switch (choice)
    {
        case "1":
            TrainAndSaveModel();
            break;

        case "2":
            LoadModelAndPredict();
            break;

        case "0":
            Console.WriteLine("Завершення програми.");
            return;

        default:
            Console.WriteLine("Невірний вибір. Спробуйте ще раз.");
            break;
    }
}

void TrainAndSaveModel()
{
    IDataView data = mlContext.Data.LoadFromTextFile<StockData>(
        path: dataPath,
        hasHeader: true,
        separatorChar: ',');

    DataOperationsCatalog.TrainTestData splitData =
        mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

    var pipeline = mlContext.Transforms.Concatenate(
            "Features",
            nameof(StockData.Open),
            nameof(StockData.High),
            nameof(StockData.Low),
            nameof(StockData.Volume))
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        
        .Append(mlContext.Regression.Trainers.FastTree(          //Sdca || FastTree || OnlineGradientDescent
            labelColumnName: nameof(StockData.Close),
            featureColumnName: "Features",
            numberOfLeaves: 20, //к-сть листків у дереві
            numberOfTrees: 100, // к-сть дерев
            minimumExampleCountPerLeaf: 10, // к-сть прикладів
            learningRate: 0.2)); //швидкість навчання 


        /*    
        
        .Append(mlContext.Regression.Trainers.OnlineGradientDescent( // Підключення алгоритму для навчання 
            labelColumnName: nameof(StockData.Close), // Що алгоритм вчиться передбачати
            featureColumnName: "Features")); // На основі яких даних алгоритм вчиться передбачати 

        */

    Console.WriteLine("Початок тренування моделі...");

    ITransformer model = pipeline.Fit(splitData.TrainSet);

    Console.WriteLine("Тренування завершено.");

    mlContext.Model.Save(model, splitData.TrainSet.Schema, modelPath);

    Console.WriteLine($"Модель збережено тут: {Path.GetFullPath(modelPath)}");

    IDataView transformedData = model.Transform(splitData.TestSet);

    RegressionMetrics metrics = mlContext.Regression.Evaluate(
        transformedData,
        labelColumnName: nameof(StockData.Close),
        scoreColumnName: "Score");

    Console.WriteLine();
    Console.WriteLine("Оцінка якості моделі:");
    Console.WriteLine($"R²:   {metrics.RSquared:0.####}");
    Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:0.####}");
    Console.WriteLine($"MSE:  {metrics.MeanSquaredError:0.####}");
    Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.####}");

    PredictWithModel(model);
}

void LoadModelAndPredict()
{
    if (!File.Exists(modelPath))
    {
        Console.WriteLine("Файл model.zip не знайдено. Спочатку натренуйте модель.");
        return;
    }

    ITransformer loadedModel = mlContext.Model.Load(modelPath, out DataViewSchema modelSchema);

    Console.WriteLine($"Модель завантажено з файлу: {Path.GetFullPath(modelPath)}");

    PredictWithModel(loadedModel);
}

void PredictWithModel(ITransformer model)
{

    PredictionEngine<StockData, StockPrediction> predictionEngine =
        mlContext.Model.CreatePredictionEngine<StockData, StockPrediction>(model);
/*
    StockData sample = new StockData
    {
        Open = 20f,
        High = 30f,
        Low = 19f,
        Volume = 3000000f
    };
*/
    StockData sample = ReadStockDataFromConsole();
    StockPrediction prediction = predictionEngine.Predict(sample);

    Console.WriteLine();
    Console.WriteLine("Приклад прогнозу:");
    Console.WriteLine($"Open:   {sample.Open}");
    Console.WriteLine($"High:   {sample.High}");
    Console.WriteLine($"Low:    {sample.Low}");
    Console.WriteLine($"Volume: {sample.Volume}");

    Console.WriteLine($"Сформований масив Features для моделі: [{string.Join(", ", prediction.Features)}]");
    Console.WriteLine($"Прогнозована ціна Close: {prediction.ClosePrice:0.####}");
}

StockData ReadStockDataFromConsole()
{
    Console.WriteLine();
    Console.WriteLine("Введіть дані для прогнозування:");

    float open = ReadFloat("Open: ");
    float high = ReadFloat("High: ");
    float low = ReadFloat("Low: ");
    float volume = ReadFloat("Volume: ");

    return new StockData
    {
        Open = open,
        High = high,
        Low = low,
        Volume = volume
    };
}

float ReadFloat(string message)
{
    while (true)
    {
        Console.Write(message);
        string input = Console.ReadLine();

        if (float.TryParse(input, out float value))
        {
            return value;
        }

        Console.WriteLine("Помилка: введіть числове значення.");
    }
}

public class StockData
{
    [LoadColumn(1)]
    public float Open { get; set; }

    [LoadColumn(2)]
    public float High { get; set; }

    [LoadColumn(3)]
    public float Low { get; set; }

    [LoadColumn(4)]
    public float Close { get; set; }

    [LoadColumn(6)]
    public float Volume { get; set; }
}

public class StockPrediction
{
    [ColumnName("Score")]
    public float ClosePrice { get; set; }

    [ColumnName("Features")]
    public float[] Features { get; set; }
}