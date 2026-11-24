using Microsoft.ML;
using Microsoft.ML.Data;

namespace IceCreamLab;

public class SalesInput
{
    [VectorType(3)]
    public float[] Features { get; set; } = new float[3];

    public static SalesInput FromTemperature(float temperature) =>
        new() { Features = ExpandTemperature(temperature) };

    public static float[] ExpandTemperature(float temperature)
    {
        float square = temperature * temperature;
        return [temperature, square, square * temperature];
    }
}

public sealed class SalesExample : SalesInput
{
    public float Label { get; set; }
}

public sealed class NativePrediction
{
    public float Score { get; set; }
}

public sealed class OnnxPrediction
{
    [ColumnName("Score")]
    public float[] Scores { get; set; } = [];
}

public static class SalesModel
{
    public const string ModelFile = "models/ice-cream.onnx";
    public const string ModelName = "IceCreamSales";

    public static bool IsValidTemperature(float temperature) =>
        float.IsFinite(temperature) && temperature >= -5 && temperature <= 5;

    public static IDataView InputSchema(MLContext context) =>
        context.Data.LoadFromEnumerable(Array.Empty<SalesInput>());

    public static ITransformer Import(MLContext context, string modelPath)
    {
        const string output = "Score.output";
        var pipeline = context.Transforms.ApplyOnnxModel(
            inputColumnNames: [nameof(SalesInput.Features)],
            outputColumnNames: [output],
            modelFile: modelPath
        );

        return pipeline.Fit(InputSchema(context));
    }
}
