using DotNetMachineLearning.Core;
using DotNetMachineLearning.Models;
using Microsoft.ML;

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
            dataView = base.CreateFeaturesFromColumns<CreditScoringTransformed>(dataView, columns, convertToVector) as IDataView;
        }

        public IEnumerable<CreditScoringTransformed> GetTransformedData()
        {
            return MLContext.Data.CreateEnumerable<CreditScoringTransformed>(dataView, reuseRowObject: false);
        }
    }
}
