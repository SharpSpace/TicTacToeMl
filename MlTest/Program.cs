// See https://aka.ms/new-console-template for more information

using System.Numerics;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

Console.WriteLine("Hello, World!");

var gameData = new GameData();
var game = new Game();
var ai = new AI();
var player = 1;
game.Print();
while (Console.ReadKey().Key != ConsoleKey.Escape)
{
    var array = game.Moves.ToList();
    for (int i = array.Count; i < 10; i++)
    {
        array.Add(0);
    }

    var singleGameData = new SingleGameData
    {
        Move1 = array[0],
        Move2 = array[1],
        Move3 = array[2],
        Move4 = array[3],
        Move5 = array[4],
        Move6 = array[5],
        Move7 = array[6],
        Move8 = array[7],
        Move9 = array[8],
    };
    var input = 0;
    if (player == 2)
    {
        input = ai.Predict(singleGameData);
    }
        
    if (game.Input(player, input))
    {
        
    }
    game.Print();

    array = game.Moves.ToList();
    for (int i = array.Count; i < 10; i++)
    {
        array.Add(0);
    }

    singleGameData = new SingleGameData
    {
        Move1 = array[0],
        Move2 = array[1],
        Move3 = array[2],
        Move4 = array[3],
        Move5 = array[4],
        Move6 = array[5],
        Move7 = array[6],
        Move8 = array[7],
        Move9 = array[8],
    };

    gameData.SingleGameDatas.Add(singleGameData);
    gameData.Save();

    player = player == 1 ? 2 : 1;
    ai.Train();
}

public class AI
{
    private static ITransformer _trainedModel;
    private readonly MLContext _mlContext;

    public AI()
    {
        _mlContext = new MLContext();

        Train();

    }

    public void Train()
    {
        if (File.Exists("my-data-file.csv") == false)
        {
            return;
        }

        // Create data prep transformer

        var data = _mlContext.Data.LoadFromTextFile<SingleGameData>("my-data-file.csv", separatorChar: ',', hasHeader: true);

        IEstimator<ITransformer> dataPrepEstimator =
            _mlContext.Transforms.Concatenate("Features", "Move1", "Move2", "Move3", "Move4", "Move5", "Move6", "Move7", "Move8", "Move9")
                .Append(_mlContext.Transforms.NormalizeMinMax("a", "Features"))
                .Append(_mlContext.Regression.Trainers.Sdca("Move1"))
                ;

        // Create data prep transformer
        _trainedModel = dataPrepEstimator.Fit(data);
 
        //IDataView transformedTrainingData = dataPrepTransformer.Transform(data);

        //// Define StochasticDualCoordinateAscent regression algorithm estimator
        //var sdcaEstimator = _mlContext.Regression.Trainers.Sdca(labelColumnName: "Features");

        //// Build machine learning model
        //_trainedModel = sdcaEstimator.Fit(transformedTrainingData);

        _mlContext.Model.Save(_trainedModel, data.Schema, "model.zip");
    }

    public int Predict(SingleGameData singleGameData)
    {
        var predictionPipeline = _mlContext.Model.Load("model.zip", out _);
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<SingleGameData, SingleGameDataPrediction>(predictionPipeline);
        var singleGameDataPrediction = predictionEngine.Predict(singleGameData);
        return singleGameDataPrediction.PredictedInput;
    }
}

public class SingleGameData
{
    [LoadColumn(0)]
    public float Move1 { get; set; }

    [LoadColumn(1)] public float Move2 { get; set; }

    [LoadColumn(2)] public float Move3 { get; set; }

    [LoadColumn(3)] public float Move4 { get; set; }

    [LoadColumn(4)] public float Move5 { get; set; }

    [LoadColumn(5)] public float Move6 { get; set; }

    [LoadColumn(6)] public float Move7 { get; set; }

    [LoadColumn(7)] public float Move8 { get; set; }

    [LoadColumn(8)] public float Move9 { get; set; }

    //    [LoadColumn(0, 8)]
    //    [VectorType(9)]
    ////    [ColumnName("Label")]
    //    public float[] Moves { get; set; }

    //[LoadColumn(9)]
    //public float Features { get; set; }

    //[LoadColumn(10)]
    //[ColumnName("Label")]
    //public float[] Features => new []{ 1f }; // string.Join(",", Moves);
}

public class SingleGameDataPrediction
{
    [ColumnName("Score")]
    public int PredictedInput { get; set; }
}

public class GameData
{
    public List<SingleGameData> SingleGameDatas { get; set; } = new List<SingleGameData>();

    public void Save()
    {
        var stringBuilder = new StringBuilder();
        if (File.Exists("my-data-file.csv"))
        {
            stringBuilder.AppendLine(File.ReadAllText("my-data-file.csv"));
        }
        else
        {
            stringBuilder.AppendLine(string.Join(",", Enumerable.Range(1, 9).Select(x => $"Move{x}")));
        }


        foreach (var singleGameData in SingleGameDatas)
        {
            stringBuilder.Append(singleGameData.Move1);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move2);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move3);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move4);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move5);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move6);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move7);
            stringBuilder.Append(",");
            stringBuilder.Append(singleGameData.Move8);
            stringBuilder.Append(",");
            stringBuilder.AppendLine(singleGameData.Move9.ToString());
        }

        File.WriteAllText("my-data-file.csv", stringBuilder.ToString());
    }
}

public class Game
{
    public Game()
    {
        Board = new int[3][];
        for (var i = 0; i < Board.Length; i++)
        {
            Board[i] = new int[3];
            for (var i1 = 0; i1 < Board[i].Length; i1++)
            {
                Board[i][i1] = 0;
            }
        }
    }

    public HashSet<int> Moves = new HashSet<int>();

    public int[][] Board { get; set; }

    public void Print()
    {
        foreach (var x in Board)
        {
            foreach (var y in x)
            {
                Console.Write(y);
            }

            Console.WriteLine(string.Empty);
        }
    }

    public bool Input(int player, int aiInput)
    {
        int input;
        if (aiInput != 0)
        {
            input = aiInput;
            Console.WriteLine(input);
        }
        else
        {
            var key = Console.ReadKey();
            Console.WriteLine(string.Empty);
            if (!int.TryParse(key.KeyChar.ToString(), out input)) return false;
        }

        if (Moves.Contains(input) || input is < 1 or > 9)
        {
            return false;
        }

        Moves.Add(input);

        var x = input / 3;
        var y = (input % 3) - 1;
        Board[x][y] = player;

        foreach (var row in Board)
        {
            if (row.All(i => i == 1))
            {
                return true;
            }

            if (row.All(i => i == 2))
            {
                return true;
            }
        }

        return false;
    }
}