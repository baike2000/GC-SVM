using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SVM.ML.DataModels;

namespace SVM.ML.Predictors
{
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "svm.mdl");
        private readonly MLContext _mlContext;

        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(111);
        }

        /// <summary>
        /// Runs prediction on new data.
        /// </summary>
        /// <param name="newSample">New data sample.</param>
        /// <returns>An object which contains predictions made by model.</returns>
        public PedastrianBinPrediction Predict(PedastrianBinData newSample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model.
              CreatePredictionEngine<PedastrianBinData, PedastrianBinPrediction>(_model);

            return predictionEngine.Predict(newSample);
        }

        private void LoadModel()
        {
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
            }

            using (var stream = new FileStream(ModelPath,
                                         FileMode.Open,
                                     FileAccess.Read,
                                     FileShare.Read))
            {
                _model = _mlContext.Model.Load(stream, out _);
            }

            if (_model == null)
            {
                throw new Exception($"Failed to load Model");
            }
        }
    }
}
