using DotNetMachineLearning.Core.Services;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using static DotNetMachineLearning.Core.Constantes;

namespace DotNetMachineLearning.Core
{
    public abstract class PredictBase : TransformBase
    {
        private readonly IPrintMetrics printService;

        protected PredictBase(IPrintMetrics printService)
        {
            if (printService == null) throw new ArgumentNullException();
            this.printService = printService;
        }

        protected virtual ColumnConcatenatingTransformer TrainModel(object dataView, string[] features, MachineLearningModel machineLearningModel, string modelNameSaved)
        {
            var data = dataView as IDataView;
            var pipeline = MLContext.Transforms.Concatenate("Features", features);
            switch (machineLearningModel)
            {
                case MachineLearningModel.Regression:
                    pipeline.Append(MLContext.Regression.Trainers.Sdca());
                    break;
                case MachineLearningModel.MulticlassClassification:
                    pipeline.Append(MLContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy());
                    break;
                case MachineLearningModel.BinaryClassification:
                    pipeline.Append(MLContext.BinaryClassification.Trainers.SdcaLogisticRegression());
                    break;
                case MachineLearningModel.SVMBinaryClassification:
                    pipeline.Append(MLContext.BinaryClassification.Trainers.LinearSvm());
                    break;
                case MachineLearningModel.LightGBMBinaryClassification:
                    var optionsForBinaryClass = new LightGbmBinaryTrainer.Options
                    {
                        Booster = new GossBooster.Options
                        {
                            TopRate = 0.3,
                            OtherRate = 0.2
                        }
                    };
                    pipeline.Append(MLContext.BinaryClassification.Trainers.LightGbm(optionsForBinaryClass));
                    break;
                case MachineLearningModel.LightGBMBinaryMultiClassClassification:
                    var optionsForMultiClass = new LightGbmMulticlassTrainer.Options
                    {
                        Booster = new DartBooster.Options()
                        {
                            TreeDropFraction = 0.15,
                            XgboostDartMode = false
                        }
                    };
                    pipeline.Append(MLContext.MulticlassClassification.Trainers.LightGbm(optionsForMultiClass));
                    break;
                default:
                    break;
            }                

            // Train model
            var trainedModel = pipeline.Fit(data);
            // Save model
            MLContext.Model.Save(trainedModel, data.Schema, $"{modelNameSaved}.zip");

            return trainedModel;
        }

        public virtual object EvaluateModel(object dataView, MachineLearningModel machineLearningModel, string modelNameSaved)
        {
            var trainedModel = MLContext.Model.Load($"{modelNameSaved}.zip", out var modelSchema);
            return null;
        }

        public virtual object Predict(object dataView, MachineLearningModel machineLearningModel, string modelNameSaved)
        {
            var trainedModel = MLContext.Model.Load($"{modelNameSaved}.zip", out var modelSchema);
            return null;
        }

        public void PrintMetrics(object metrics)
        {
            printService.Print(metrics);
        }

    }
}
