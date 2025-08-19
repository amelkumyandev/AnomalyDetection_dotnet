using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace DdosAutoencoder.Utils;

/// <summary>Classic z-score scaler with JSON (de)serialisation.</summary>
public sealed record StandardScaler(double[] Mean, double[] Std)
{
    public static StandardScaler Fit(IList<double[]> data)
    {
        int dim = data[0].Length;
        var mean = new double[dim];
        var var  = new double[dim];

        /* exact mean */
        foreach (var v in data)
            for (int j = 0; j < dim; j++)
                mean[j] += v[j];
        for (int j = 0; j < dim; j++) mean[j] /= data.Count;

        /* exact variance Σ(x−μ)² / N */
        foreach (var v in data)
            for (int j = 0; j < dim; j++)
            {
                double d = v[j] - mean[j];
                var[j] += d * d;
            }

        const double eps = 1e-6;                 // keep denominator ≥ eps
        var std = new double[dim];
        for (int j = 0; j < dim; j++)
            std[j] = Math.Sqrt(var[j] / data.Count) + eps;

        return new StandardScaler(mean, std);
    }


    public double[] Transform(double[] v)
    {
        var res = new double[v.Length];
        for (int i = 0; i < v.Length; i++)
            res[i] = (v[i] - Mean[i]) / Std[i];
        return res;
    }

    public Tensor Transform(double[] v, ScalarType dtype = ScalarType.Float32) =>
        Transform(v).ToTensor(dtype);

    public IList<double[]> Transform(IEnumerable<double[]> rows) =>
        rows.Select(Transform).ToList();

    public void Save(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(this));

    public static StandardScaler Load(string path) =>
        JsonSerializer.Deserialize<StandardScaler>(File.ReadAllText(path))
        ?? throw new InvalidOperationException("Failed to load scaler");
}