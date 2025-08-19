using CsvHelper;
using CsvHelper.Configuration;
using DdosAutoencoder.Utils;
using System.Globalization;

namespace DdosAutoencoder.Training;

public static class CsvLoader
{
    private static readonly string[] SkipCols =
    {
        "FLOW ID", "SOURCE IP", "DESTINATION IP", "TIMESTAMP"
    };

    public static (List<double[]> X, List<int> y, string[] header) Load(string csvPath)
    {
        using var reader = new StreamReader(csvPath);
        using var csv    = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            PrepareHeaderForMatch = a => a.Header.Trim(),   // removes leading space
            MissingFieldFound     = null,
            BadDataFound          = null
        });

        csv.Read(); csv.ReadHeader();
        var rawHeader = csv.HeaderRecord!;
        var header    = rawHeader.Select(h => h.Trim()).ToArray();   // normalised copy

        /* ── numeric columns (skip IDs & label) ───────────────────────── */
        var numericIdx = header
            .Select((h, i) => (h, i))
            .Where(t => !SkipCols.Contains(t.h.ToUpperInvariant())
                        && t.h.ToUpperInvariant() != "LABEL")
            .Select(t => t.i)
            .ToArray();

        /* ── label column index (robust) ──────────────────────────────── */
        int labelIdx = Array.FindIndex(header,
            h => h.Equals("LABEL", StringComparison.OrdinalIgnoreCase));

        if (labelIdx == -1)           // fallback: assume last column is label
            labelIdx = header.Length - 1;

        /* ── helpers ---------------------------------------------------- */
        static double ParseCell(string cell)
        {
                 if (!double.TryParse(cell, NumberStyles.Any, CultureInfo.InvariantCulture, out var v))
                         return 0.0;
               return double.IsFinite(v) ? v : 0.0;          // strip ±Infinity and NaN
        }

        var features = new List<double[]>();
        var labels   = new List<int>();

        while (csv.Read())
        {
            var record = csv.Parser.Record!;

            var row = new double[numericIdx.Length];
            for (int j = 0; j < numericIdx.Length; j++)
            {
                int idx = numericIdx[j];
                row[j] = idx < record.Length ? ParseCell(record[idx]) : 0.0;
            }
            features.Add(row);

            string labelCell = labelIdx < record.Length ? record[labelIdx] : "BENIGN";
            labels.Add(labelCell.Trim().Equals("BENIGN", StringComparison.OrdinalIgnoreCase) ? 0 : 1);
        }

        return (features, labels, header);
    }
}
