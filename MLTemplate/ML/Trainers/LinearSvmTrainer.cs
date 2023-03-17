using Microsoft.ML.Trainers;
using Microsoft.ML;
using SVM.ML.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace SVM.ML.Trainers
{
    public class LinearSvmTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public LinearSvmTrainer() : base()
        {
            Name = "Linear SVM";
            _model = MlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label");
        }
    }
}
