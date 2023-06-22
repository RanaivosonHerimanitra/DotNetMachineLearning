using Microsoft.ML;
using Microsoft.ML.Data;
using static DotNetMachineLearning.Core.Constantes;

namespace DotNetMachineLearning.Core
{
    public abstract class PredictBase : TransformBase
    {
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
                case MachineLearningModel.SVM:
                    break;
                case MachineLearningModel.LightGBM:
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

        public virtual object Predict(object dataView,MachineLearningModel machineLearningModel, string modelNameSaved)
        {
            var trainedModel = MLContext.Model.Load($"{modelNameSaved}.zip", out var modelSchema);
            return null;
        }
    }
}
