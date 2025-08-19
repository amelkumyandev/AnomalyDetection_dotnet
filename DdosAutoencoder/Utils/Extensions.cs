using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;

namespace DdosAutoencoder.Utils;

public static class Extensions
{
    /* ---------- tensor helpers ---------- */

    /// <summary>Convert list of rows (double[]) to 2-D Float32 tensor.</summary>
    public static Tensor ToTensor(this IList<double[]> data)
    {
        long rows = data.Count;
        long cols = data[0].Length;
        var flat  = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            flat[i * cols + j] = (float)data[i][j];

        return torch.tensor(flat, new long[] { rows, cols }, dtype: ScalarType.Float32);
    }

    public static Tensor ToTensor(this double[] v, ScalarType dtype = ScalarType.Float32)
        => torch.tensor(v.Select(d => (float)d).ToArray(), new long[] { v.Length }, dtype: dtype);

    /* ---------- stats helpers ---------- */

    public static double Mean(this IEnumerable<double> seq) => seq.Average();

    public static double StdDev(this IEnumerable<double> seq)
    {
        var arr = seq.ToArray();
        double mu = arr.Average();
        return Math.Sqrt(arr.Sum(x => Math.Pow(x - mu, 2)) / arr.Length);
    }

    /* ---------- parsing helper ---------- */

    public static double ParseInvariant(this string s) =>
        double.Parse(s, NumberStyles.Any, CultureInfo.InvariantCulture);
}