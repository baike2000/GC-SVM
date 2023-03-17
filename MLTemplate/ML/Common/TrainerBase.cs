using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SVM.ML.DataModels;

namespace SVM.ML.Common
{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name.
    /// </summary>
    public abstract class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path
                          .Combine(AppContext.BaseDirectory, "svm.mdl");

        protected readonly MLContext MlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            MlContext = new MLContext(111);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>Model performance.</returns>
        public BinaryClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return MlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
        }

        /// <summary>
        /// Save Model in the file.
        /// </summary>
        public void Save()
        {
            MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        /// <summary>
        /// Feature engeneering and data pre-processing.
        /// </summary>
        /// <returns>Data Processing Pipeline.</returns>
        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = MlContext.Transforms.Concatenate("Features",
                                               nameof(PedastrianBinData.Hog)
                                               )
               .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
               .AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data
                                    .LoadFromTextFile<PedastrianBinData>
                                      (trainingFileName, hasHeader: true, separatorChar: ',');
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }

}
