using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext();

IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
    path: "/path/to/dataset.tsv",
    hasHeader: true,
    separatorChar: '\t');

DataOperationsCatalog.TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainingData = trainTestSplit.TrainSet;
IDataView testData = trainTestSplit.TestSet;

var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText));

var trainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
var trainingPipeline = dataProcessPipeline.Append(trainer);

ITransformer trainedModel = trainingPipeline.Fit(trainingData);

var predictions = trainedModel.Transform(testData);
var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
Console.WriteLine($"F1 score: {metrics.F1Score}");

var engine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(trainedModel);

var testReview = new SentimentData { SentimentText = "good movie" };
var result = engine.Predict(testReview);

Console.WriteLine($"Predicted sentiment: {(result.Prediction ? "Positive" : "Negative")}");
Console.WriteLine($"Probability: {result.Probability}");
Console.WriteLine($"Score: {result.Score}");

public class SentimentData
{
    [LoadColumn(0)]
    public string SentimentText;

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment;
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}