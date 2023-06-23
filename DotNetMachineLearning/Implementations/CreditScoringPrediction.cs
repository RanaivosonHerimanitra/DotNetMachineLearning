﻿using DotNetMachineLearning.Core;
using DotNetMachineLearning.Core.Services;
using DotNetMachineLearning.Models;
using Microsoft.ML;
using static DotNetMachineLearning.Core.Constantes;

namespace DotNetMachineLearning
{
    public sealed class CreditScoringPrediction : PredictBase
    {
        private IDataView dataView;
        private IPrintMetrics printMetrics;

        public CreditScoringPrediction(IEnumerable<CreditScoring> data, IPrintMetrics printMetrics): base(printMetrics)
        {
            dataView = MLContext.Data.LoadFromEnumerable(data);
            this.printMetrics = printMetrics;
        }

        public void CreateFeaturesFromColumns(Dictionary<string, string> columns, bool convertToVector)
        {
            try
            {
                dataView = (IDataView)base.CreateFeaturesFromColumns<CreditScoringTransformed>(dataView, columns, convertToVector);

            }
            catch (InvalidCastException)
            {

                throw new InvalidCastException(Constantes.DoitEtreDataView);
            }
        }

        public IEnumerable<CreditScoringTransformed> GetTransformedData()
        {
            return MLContext.Data.CreateEnumerable<CreditScoringTransformed>(dataView, false);
        }

        public void TrainModel(string[] features, MachineLearningModel machineLearningModel, string modelName)
        {
           base.TrainModel(dataView, features, machineLearningModel, modelName);
        }
    }
}
