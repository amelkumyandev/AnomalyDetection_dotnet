### Quick start

```bash
# 1  Download dataset slice into Data/
wget -P src/DdosAutoencoder/Data \
  https://www.unb.ca/cic/datasets/ddos-2019/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

# 2  Train
dotnet run --project src/DdosAutoencoder -- train

# 3  Serve
dotnet run --project src/DdosAutoencoder -- serve 5000
