using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM.ML.Common
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        BinaryClassificationMetrics Evaluate();
        void Save();
    }
}
