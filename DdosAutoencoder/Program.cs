using DdosAutoencoder.Training;

// Default dataset path (relative to project root)
const string defaultCsv = "Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv";

var csv = args.Length > 0 ? args[0] : defaultCsv;
Trainer.Run(csv);