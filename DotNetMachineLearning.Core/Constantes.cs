
namespace DotNetMachineLearning.Core
{
    public static class Constantes
    {
        public enum MachineLearningModel
        {
             Regression = 1,
             MulticlassClassification = 2,
             BinaryClassification = 3,
             SVMBinaryClassification = 4,
             LightGBMBinaryClassification = 5,
             LightGBMBinaryMultiClassClassification = 6
        };

        public static string DelimiteurDoitExister = "Le délimiteur doit etre défini.";

        public static string DoitEtreDataView = "La conversion en dataview ne s'est pas bien passé. Vérifiez vos données.";

        public static string FichierDoitExister = "Le ou les fichiers doivent etre spécifiés.";
    }
}
