// -----------------------------------------------------------------------------
//  Program.cs – Hyper-parameter search (PCA rank & K) with Silhouette + DBI
// -----------------------------------------------------------------------------
#pragma warning disable CA1852

using System.Globalization;
using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;

#region  POCOs
public sealed class FlowInput
{
    [VectorType(77)] public float[] Features { get; set; } = default!;
    public string    Label          { get; set; } = default!;
}

/// <summary> Cluster assignment + centroid-distance vector (Score).</summary>
public sealed class ClusterOut
{
    [ColumnName("PredictedLabel")] public uint   ClusterId { get; set; }      // 1-based
    [VectorType]                  public float[] Score     { get; set; } = default!;
}
#endregion

internal static class Program
{
    // Candidate hyper-parameters ------------------------------------------------
    private static readonly int[]   PcaRanks  = { 25, 30 };
    private static readonly int[]   KValues   = {  8, 10, 12 };
    private const int               Seed      = 42;

    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    private static readonly string      Path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv";

    public static void Main()
    {
        var ml   = new MLContext(seed: Seed);
        Console.Write("Loading CSV … ");
        var data = ml.Data.LoadFromEnumerable(StreamCsv(Path));
        Console.WriteLine("done");

        // Keep results for best model ------------------------------------------
        (double Sil, double Dbi, int Rank, int K, double Acc, double Dr, double Far) best =
            (double.MinValue, double.MaxValue, 0, 0, 0, 0, 0);

        foreach (int rank in PcaRanks)
        foreach (int k     in KValues)
        {
            Console.Write($"Training  (rank={rank,2}, k={k,2}) … ");

            // ── pipeline ───────────────────────────────────────────────────────
            var pipe = ml.Transforms.ReplaceMissingValues("Features", "Features",
                                replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(ml.Transforms.NormalizeMeanVariance("Features"))
                .Append(ml.Transforms.ProjectToPrincipalComponents("Pca", "Features", rank: rank))
                .Append(ml.Clustering.Trainers.KMeans(new KMeansTrainer.Options
                {
                    FeatureColumnName         = "Pca",
                    NumberOfClusters          = k,
                    InitializationAlgorithm   = KMeansTrainer.InitializationAlgorithm.KMeansPlusPlus,
                    MaximumNumberOfIterations = 500,
                    OptimizationTolerance     = 1e-4f,
                }));

            var model = pipe.Fit(data);
            var preds = model.Transform(data);

            // ── built-in cluster metrics (DBI) ────────────────────────────────
            var dbi = ml.Clustering.Evaluate(data: preds, scoreColumnName:"Score", featureColumnName: "Features").DaviesBouldinIndex;

            // ── custom metrics (Silhouette + accuracy) ─────────────────────────
            var scored = ml.Data.CreateEnumerable<ClusterOut>(preds, false).ToArray();
            var labels = ml.Data.CreateEnumerable<FlowInput>(data,   false)
                                .Select(r => r.Label == "BENIGN" ? 0 : 1).ToArray();

            double sil = ApproxSilhouette(scored);                   // ← new
            (double acc, double dr, double far) = ConfMatrix(scored, labels);

            Console.WriteLine($"Sil {sil:0.000} | DBI {dbi:0.000} | Acc {acc:P1}");

            // keep best (highest Silhouette, then lowest DBI) ------------------
            if (sil > best.Sil || (Math.Abs(sil - best.Sil) < 1e-6 && dbi < best.Dbi))
                best = (sil, dbi, rank, k, acc, dr, far);
        }

        // ---------------------------------------------------------------------
        Console.WriteLine("\n=== BEST CONFIGURATION ===============================");
        Console.WriteLine($"PCA rank          : {best.Rank}");
        Console.WriteLine($"Clusters (k)      : {best.K}");
        Console.WriteLine($"Silhouette score  : {best.Sil:0.000}");
        Console.WriteLine($"Davies-BouldinIdx : {best.Dbi:0.000}");
        Console.WriteLine($"Accuracy          : {best.Acc:P2}");
        Console.WriteLine($"Detection rate    : {best.Dr:P2}");
        Console.WriteLine($"False-alarm rate  : {best.Far:P2}");
    }

    // ───────────────────────────────────────────────────────────────────────────
    private static double ApproxSilhouette(IEnumerable<ClusterOut> rows)
    {
        double sum = 0; int n = 0;
        foreach (var r in rows)
        {
            int   id = (int)r.ClusterId - 1;          // 0-based
            float a  = r.Score[id];                   // distance to own centroid
            float b  = r.Score.Where((_,i) => i!=id).Min(); // nearest other centroid
            double s = (b - a) / Math.Max(a, b);
            sum += s; n++;
        }
        return sum / n;
    }

    private static (double Acc, double DetRate, double FarRate) ConfMatrix(
        ClusterOut[] clusters, int[] labels)
    {
        // Map cluster → majority label (0 = benign, 1 = malicious) ------------
        var map = clusters.Zip(labels)
                          .GroupBy(z => z.First.ClusterId)
                          .ToDictionary(g => g.Key,
                                        g => g.GroupBy(z => z.Second)
                                              .OrderByDescending(h => h.Count())
                                              .First().Key);

        int tp=0, tn=0, fp=0, fn=0;
        for (int i = 0; i < clusters.Length; i++)
        {
            int pred  = map[clusters[i].ClusterId];
            int truth = labels[i];
            if (pred==1 && truth==1) tp++;
            else if (pred==0 && truth==0) tn++;
            else if (pred==1 && truth==0) fp++;
            else fn++;
        }
        double total = tp+tn+fp+fn;
        return ((tp+tn)/total,
                tp/(double)(tp+fn),
                fp/(double)(fp+tn));
    }

    /// Skip first 7 metadata columns, use next 77 as features, last as label.
    private static IEnumerable<FlowInput> StreamCsv(string path)
    {
        const int meta = 7;
        using var reader = new StreamReader(path);
        using var csv    = new CsvReader(reader, Inv);

        csv.Read(); csv.ReadHeader();
        int last = csv.HeaderRecord!.Length - 1;

        while (csv.Read())
        {
            var vec = new float[last - meta];
            for (int i = meta; i < last; i++)
            {
                string s = csv.GetField(i);
                if (!float.TryParse(s, NumberStyles.Float, Inv, out float v) ||
                     float.IsNaN(v) || float.IsInfinity(v))
                    v = 0f;
                vec[i - meta] = v;
            }
            yield return new FlowInput { Features = vec, Label = csv.GetField(last) };
        }
    }
}
