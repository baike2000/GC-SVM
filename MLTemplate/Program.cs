using SVM.ML.Common;
using SVM.ML.DataModels;
using SVM.ML.Predictors;
using SVM.ML.Trainers;
using System;
using System.Collections.Generic;

namespace SVM
{
    class Program
    {

        static void Main(string[] args)
        {
            // Новый пример того, что надо распознать (нужны реальные данные)
            var newSample = new PedastrianBinData
            {
                Hog = new List<float>() { 1.0f, 2.0f, 3.0f } // Тут должно быть 1024 параметра
            };

            // Различные модели для обучения
            var trainers = new List<ITrainerBase>
            {
                new LdSvmTrainer(3),
                new LdSvmTrainer(5),
                new LdSvmTrainer(7),
                new LinearSvmTrainer()
            };

            trainers.ForEach(t => TrainEvaluatePredict(t, newSample));
        }

        static void TrainEvaluatePredict(ITrainerBase trainer, PedastrianBinData newSample)
        {
            Console.WriteLine("*******************************");
            Console.WriteLine($"{trainer.Name}");
            Console.WriteLine("*******************************");

            trainer.Fit("pedastrian_binary.csv"); // Данные для обучения - нужно подготовить.

            var modelMetrics = trainer.Evaluate();
            // ROC кривая
            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:0.##}{Environment.NewLine}" +
                              $"F1 Score: {modelMetrics.F1Score:#.##}{Environment.NewLine}" +
                              $"Positive Precision: {modelMetrics.PositivePrecision:#.##}{Environment.NewLine}" +
                              $"Negative Precision: {modelMetrics.NegativePrecision:0.##}{Environment.NewLine}" +
                              $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                              $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}" +
                              $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:#.##}{Environment.NewLine}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(newSample); // прдесказание
            Console.WriteLine("------------------------------");
            Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
            Console.WriteLine("------------------------------");
        }
    }
}