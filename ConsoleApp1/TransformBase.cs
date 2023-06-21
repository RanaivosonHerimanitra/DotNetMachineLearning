using Microsoft.ML;

namespace DotNetMachineLearning
{
    public abstract class TransformBase: ExtractBase
    {
        /// <summary>
        /// ColumnName is a dictionary of Original column (Key)
        /// and the newly created column vector that corresponding to it (Value).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="columnName"></param>
        /// <returns></returns>
        protected virtual object CreateFeaturesFromColumns<T>(IDataView data, Dictionary<string, string> columnName, bool convertToVector)
        {
            var inputOutputColumnPairs = columnName.Select(x => new InputOutputColumnPair(x.Key, x.Value)).ToArray();
            if (convertToVector)
            {
                var vectorTransformation = MLContext.Transforms.Conversion.MapKeyToVector(inputOutputColumnPairs);
                return vectorTransformation.Fit(data).Transform(data);
            }
            var numericalTransformation = MLContext.Transforms.Conversion.MapValueToKey(inputOutputColumnPairs);
            return numericalTransformation.Fit(data).Transform(data);
        }
    }
}
