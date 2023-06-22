using DotNetMachineLearning.Core;
using DotNetMachineLearning.Models;
using Microsoft.ML;
using static DotNetMachineLearning.Core.Constantes;

namespace DotNetMachineLearning
{
    public class CreditScoringPrediction : PredictBase
    {
        private IDataView dataView;

        public CreditScoringPrediction(IEnumerable<CreditScoring> data)
        {
            dataView = MLContext.Data.LoadFromEnumerable(data);
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
