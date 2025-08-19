using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using DdosAutoencoder.Models;
using DdosAutoencoder.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DdosAutoencoder.Training;

public static class Trainer
{
    /* ---------------- hyper-parameters ---------------- */
    private const int    Epochs      = 100;
    private const int    BatchSize   = 512;
    private const double LrInit      = 1e-3;
    private const double TestPct     = 0.10;   // 10 % of all rows
    private const double ValPct      = 0.10;   // 10 % of benign rows
    private const int    Patience    = 3;
    private const double MinDelta    = 1e-4;

    public static void Run(string csvPath)
    {
        Console.WriteLine($"Loading {csvPath} …");
        var (features, labels, _) = CsvLoader.Load(csvPath);
        int total = features.Count;

        /* ----------- split train / val / test ----------- */
        var rnd   = new Random(42);
        var all   = Enumerable.Range(0, total).OrderBy(_ => rnd.Next()).ToArray();
        int testN = (int)(total * TestPct);
        var testI = all[..testN];
        var rest  = all[testN..];

        var benignRest = rest.Where(i => labels[i] == 0).ToArray();
        int valN       = (int)(benignRest.Length * ValPct);
        var valI       = benignRest[..valN];
        var trainI     = benignRest[valN..];

        var train = trainI.Select(i => features[i]).ToList();
        var val   = valI  .Select(i => features[i]).ToList();
        var test  = testI .Select(i => (features[i], labels[i])).ToList();

        Console.WriteLine($"Train: {train.Count} | Val: {val.Count} | Test: {test.Count}");

        /* ---------------- scaling ---------------- */
        var scaler = StandardScaler.Fit(train);
        var Xtrain = scaler.Transform(train).ToTensor();
        var Xval   = scaler.Transform(val)  .ToTensor();

        /* ---------------- model ------------------ */
        var device = cuda.is_available() ? torch.device("cuda") : torch.device("cpu");
        var ae     = new Autoencoder((int)Xtrain.shape[1]).to(device);
        var opt    = optim.Adam(ae.parameters(), lr: LrInit);
        var sched  = optim.lr_scheduler.ReduceLROnPlateau(
                       opt, mode:"min", factor:0.5, patience:3,
                       threshold:MinDelta, threshold_mode:"abs");
        var lossFn = MSELoss();

        /* ---------------- training ---------------- */
        double bestVal = double.PositiveInfinity;
        int    badEp   = 0;
        long   batches = (long)Math.Ceiling(Xtrain.shape[0] / (double)BatchSize);

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            ae.train();
            double epochLoss = 0;

            for (long b = 0; b < batches; b++)
            {
                long start = b * BatchSize;
                long len   = Math.Min(BatchSize, Xtrain.shape[0] - start);
                var  batch = Xtrain.narrow(0, start, len).to(device);

                opt.zero_grad();
                var loss = lossFn.forward(ae.forward(batch), batch);
                loss.backward();
                opt.step();

                epochLoss += loss.item<float>();
            }

            /* ---- validation ---- */
            ae.eval();
            var valRecon = ae.forward(Xval.to(device)).cpu();
            double valLoss = lossFn.forward(valRecon, Xval).cpu().item<float>();
            sched.step(valLoss);

            Console.WriteLine($"E{epoch:3}  train={epochLoss/batches:E4}  val={valLoss:E4}");

            if (bestVal - valLoss > MinDelta)
            {
                bestVal = valLoss;
                badEp   = 0;
                ae.save("Model/best_state.pt");
            }
            else if (++badEp >= Patience)
            {
                Console.WriteLine($"⏹ Early stop @ epoch {epoch}");
                break;
            }
        }

        /* ---------- reload best checkpoint ---------- */
        ae.load("Model/best_state.pt");
        ae.eval();

        /* ---------- threshold grid on val ---------- */
        var valErr = ae.forward(Xval.to(device)).cpu()
                       .sub(Xval).pow(2).mean(new long[] { 1 })
                       .cpu().data<float>().Select(f => (double)f).ToArray();
        Array.Sort(valErr);

        double bestF1Val = 0, tau = valErr[^1];
        for (double p = 0.90; p <= 0.999; p += 0.001)
        {
            double t = valErr[(int)(valErr.Length * p)];
            var (tpTmp, fpTmp, fnTmp, _) = ConfMatrix(test, t, scaler, ae, device);
            double pr = tpTmp + fpTmp == 0 ? 0 : tpTmp / (double)(tpTmp + fpTmp);
            double rc = tpTmp + fnTmp == 0 ? 0 : tpTmp / (double)(tpTmp + fnTmp);
            double f1Tmp = pr + rc == 0 ? 0 : 2 * pr * rc / (pr + rc);
            if (f1Tmp > bestF1Val) { bestF1Val = f1Tmp; tau = t; }
        }
        Console.WriteLine($"Best τ (val) = {tau:E4}  →  F1_val = {bestF1Val:P2}");

        /* ---------- evaluate on test ---------- */
        var (tp, fp, fn, tn) = ConfMatrix(test, tau, scaler, ae, device);

        double acc  = (tp + tn) / (double)test.Count;
        double prec = tp + fp == 0 ? 0 : tp / (double)(tp + fp);
        double rec  = tp + fn == 0 ? 0 : tp / (double)(tp + fn);
        double f1   = prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec);

        Console.WriteLine($$"""
        ---------------  EVALUATION  ---------------
        Total samples   : {{test.Count}}
        Accuracy        : {{acc:P2}}
        Precision       : {{prec:P2}}
        Recall          : {{rec:P2}}
        F1-Score        : {{f1:P2}}
        ConfusionMatrix : TP={{tp}}, FP={{fp}}, FN={{fn}}, TN={{tn}}
        --------------------------------------------
        """);

        /* ---------- save artefacts ---------- */
        Directory.CreateDirectory("Model");
        ae.save("Model/ae.pt");

        var jsonOpt = new System.Text.Json.JsonSerializerOptions
        {
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling
                             .AllowNamedFloatingPointLiterals
        };
        File.WriteAllText("Model/threshold.json",
            System.Text.Json.JsonSerializer.Serialize(new { threshold = tau }, jsonOpt));
        scaler.Save("Model/scaler.json");
    }

    /* ---------- helper : confusion matrix ---------- */
    private static (int tp,int fp,int fn,int tn) ConfMatrix(
        IList<(double[] x,int y)> data, double tau, StandardScaler scaler,
        Autoencoder ae, Device device)
    {
        int tp=0,fp=0,fn=0,tn=0;
        foreach (var (raw,y) in data)
        {
            var x = scaler.Transform(raw).ToTensor().unsqueeze(0).to(device);
            double e = (ae.forward(x)-x).pow(2).mean().cpu().item<float>();
            bool an  = e > tau;
            if (an && y==1) tp++;
            else if (an)    fp++;
            else if (y==1)  fn++;
            else            tn++;
        }
        return (tp,fp,fn,tn);
    }

    /* overload for pure-benign validation rows */
    private static (int tp,int fp,int fn,int tn) ConfMatrix(
        IList<double[]> benign, double tau, StandardScaler scaler,
        Autoencoder ae, Device device)
    {
        int fp=0, tn=0;
        foreach (var raw in benign)
        {
            var x = scaler.Transform(raw).ToTensor().unsqueeze(0).to(device);
            double e = (ae.forward(x)-x).pow(2).mean().cpu().item<float>();
            if (e > tau) fp++; else tn++;
        }
        return (0,fp,0,tn);
    }
}
