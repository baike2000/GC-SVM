using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVM.ML.DataModels
{
    public class PedastrianBinData
    {
        /// <summary>
        /// Models Pedastrian Binary Data.
        /// </summary>
        [LoadColumn(1)]
        public bool Label { get; set; }
        [LoadColumn(2-17)]
        public List<float> Hog { get; set; }
    }
}
