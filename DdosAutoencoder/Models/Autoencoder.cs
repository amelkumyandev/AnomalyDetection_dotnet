using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DdosAutoencoder.Models;

/// <summary>FC auto-encoder 56-256-128-32-128-256-56.</summary>
public sealed class Autoencoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _net;

    public Autoencoder(int inputDim) : base(nameof(Autoencoder))
    {
        _net = Sequential(
            ("enc1", Linear(inputDim, 256)), ("relu1", ReLU()),
            ("enc2", Linear(256, 128)),      ("relu2", ReLU()),
            ("enc3", Linear(128,  32)),      ("relu3", ReLU()),

            ("dec1", Linear( 32, 128)),      ("relu4", ReLU()),
            ("dec2", Linear(128, 256)),      ("relu5", ReLU()),
            ("out",  Linear(256, inputDim))
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => _net.forward(x);
}