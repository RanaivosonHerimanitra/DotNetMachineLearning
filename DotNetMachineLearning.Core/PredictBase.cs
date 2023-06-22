using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Data.SqlClient;

namespace DotNetMachineLearning.Core
{
    public abstract class PredictBase : TransformBase
    {
        protected virtual object TrainModel(object dataView, string[] features, RegressionCatalog regressionCatalog, string modelNameSaved)
        {
            var data = dataView as IDataView;
            var pipelineEstimator = MLContext.Transforms
                .Concatenate("Features", features)
                .Append(regressionCatalog.Trainers.Sdca());

            // Train model
            var trainedModel = pipelineEstimator.Fit(data);
            // Save model
            MLContext.Model.Save(trainedModel, data.Schema, $"{modelNameSaved}.zip");

            return trainedModel;
        }

        public virtual object Predict(string modelNameSaved)
        {
            return null;
        }
    }
}
