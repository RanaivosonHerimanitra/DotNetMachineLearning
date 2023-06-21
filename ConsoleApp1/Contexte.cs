using Microsoft.ML;

namespace DotNetMachineLearning
{
    /// <summary>
    /// Inspiration: https://www.c-sharpcorner.com/UploadFile/8911c4/singleton-design-pattern-in-C-Sharp/
    /// </summary>
    public class Contexte
    {
        private static readonly MLContext mlContext = new MLContext();

        private static MLContext instance = null;

        public Contexte()
        {
        }
        
        public static MLContext MLContext
        {
            get
            {
                if (instance == null)
                {
                    lock (mlContext)
                    {
                        instance ??= new MLContext();
                    }
                }
                return instance;
            }
        }
    }
}
