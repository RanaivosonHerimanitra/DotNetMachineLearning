using Microsoft.ML.Data;

namespace DotNetMachineLearning.Models
{
    public class CreditScoringTransformed : CreditScoring
    {
        public float[] AgeVector { get; set; }

    }

    public class CreditScoring
    {
        public bool IsDefault { get; set; }

        public int NumberOfCreditCards { get; set; }

        [KeyType(99)]
        public uint Age { get; set; }

        public float Score { get; set; }

    }
}
