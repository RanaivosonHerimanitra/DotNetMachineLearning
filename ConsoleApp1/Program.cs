using DotNetMachineLearning;
using Microsoft.ML;
// API
Console.WriteLine(Contexte.MLContext is MLContext && Contexte.MLContext != null);
var creditData = new[]
{
    new CreditScoring {Score = 200, Age = 15, IsDefault = true, NumberOfCreditCards = 2},
};
var creditScoringPrediction = new CreditScoringPrediction(creditData);
var dict = new Dictionary<string, string>
{
    { "AgeVector", "Age" }
};
creditScoringPrediction.CreateFeaturesFromColumns(dict, true);
Console.WriteLine("Score\t IsDefault\t Age\t AgeVector\t NumberOfCreditCards\t");
foreach (var item in creditScoringPrediction.GetTransformedData())
{
    Console.WriteLine($"\t{item.Score}\t {item.IsDefault}\t\t {item.Age}\t\t  {string.Join(',', item.AgeVector)}\t\t {item.NumberOfCreditCards}\t");
}
   
