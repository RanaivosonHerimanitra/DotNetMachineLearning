using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DotNetMachineLearning.Core.Services
{
    public class PrintMetrics : IPrintMetrics
    {
        public void Print(object metrics)
        {
            var multiclassMetrics = metrics as MulticlassClassificationMetrics;
            if (multiclassMetrics != null)
            {
                Console.WriteLine($"Macro Accuracy: {multiclassMetrics.MacroAccuracy:F2}");
                Console.WriteLine($"Log Loss: {multiclassMetrics.LogLoss:F2}");
                Console.WriteLine($"Log Loss Reduction: {multiclassMetrics.LogLossReduction:F2}\n");
                Console.WriteLine(multiclassMetrics.ConfusionMatrix.GetFormattedConfusionTable());

            }
        }
    }
}
