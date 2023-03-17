using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace SVM.Helpers
{
    static class ImageHelper
    {
        private const double RedIntense = 0.2125;
        private const double GreenIntense = 0.7154;
        private const double BlueIntense = 0.0721;
        private const int HogSize = 16;
        private const double Pi = 3.1415926535897;
        private const double Eps = 1e-8;
        private const double ApproxStep = 0.35;
        private const int ApproxOrder = 1;
        private const int XSize = 10;
        private const int YSize = 10;

        public static int[,] CalculateHog(Bitmap image)
        {
            double gradX, gradY;
            double angle;
            var coef = HogSize / (2 * Pi);
            var height = image.Height;
            var width = image.Width;
            int i, j;

            var hog = new int[width, height];
            for (i = 0; i < width; i++)
            {
                for (j = 0; j < height; j++)
                {
                    gradX = GetBrightness(image.GetPixel(CheckBound(i + 1, width), CheckBound(j, height))) -
                            GetBrightness(image.GetPixel(CheckBound(i - 1, width), CheckBound(j, height)));
                    gradY = GetBrightness(image.GetPixel(CheckBound(i, width), CheckBound(j + 1, height))) -
                            GetBrightness(image.GetPixel(CheckBound(i, width), CheckBound(j - 1, height)));
                    gradX /= 2.0;
                    gradY /= 2.0;
                    angle = Math.Atan2(gradY, gradX) + Pi;
                    if (angle > 2 * Pi)
                    {
                        angle -= Eps;
                    }
                    var k = (int)(angle * coef);
                    hog[i, j] = k;
                }
            }

            return hog;
        }
        public static double GetBrightness(Color p)
        {
            return p.R * RedIntense +
                   p.G * GreenIntense +
                   p.B * BlueIntense;
        }

        public static int CheckBound(int x, int n)
        {
            if (x >= n)
                return n - 1;
            if (x < 0)
                return 0;
            return x;
        }

        public static void AddUnitHog(int[,] hog, List<int> hist, int x0, int y0, int x1, int y1)
        {
            var h = new int[HogSize];
            int i, j;


            for (i = 0; i < HogSize; i++)
            {
                h[i] = 0;
            }

            for (i = x0; i < x1; i++)
            {
                for (j = y0; j < y1; j++)
                {
                    h[hog[i, j]]++;
                }
            }

            for (i = 0; i < HogSize; i++)
            {
                hist.Add(h[i]);
            }
        }

        public static List<int> GetHog(int[,] hog, int x0, int x1, int y0, int y1)
        {
            var hist = new List<int>();

            for (var i = x0; i < x1; i += XSize)
            {
                for (var j = y0; j < y1; j += YSize)
                {
                    AddUnitHog(hog, hist, i, j, CheckBound(i + XSize, x1), CheckBound(j + YSize, y1));
                }
            }

            return Nonlinear(hist);
        }

        public static List<int> Nonlinear(List<int> x)
        {
            var res = new List<int>(x);

            for (var i = 0; i < x.Count(); i++)
            {
                var z = x[i];
                for (var j = -ApproxOrder; j <= ApproxOrder; j++)
                {
                    if (z == 0)
                    {
                        res.Add(0);
                        res.Add(0);
                        continue;
                    }
                    var l = j * ApproxStep;
                    var k = 1 / Math.Cosh(Pi * l);
                    var v = Math.Sqrt(z * k);
                    res.Add((int)(Math.Cos(-l * Math.Log(z)) * v * 10));
                    res.Add((int)(Math.Sin(-l * Math.Log(z)) * v * 10));
                }
            }
            return res;
        }
    }
}
