# ERGO-II
We have previously published [ERGO](https://github.com/louzounlab/ERGO), a deep learning based method
for predicting TCR and epitope peptide binding.
Several evaluation methods were suggested, including Single Peptide Binding (SPB), Multi Peptide Selection (MPS) and
TCR Peptide Pairing (TPP).

ERGO uses only the TCRβ CDR3 sequence and the peptide sequence for the binding prediciton.
We have now developed **ERGO-II**, an updated method for TCR-peptide binding prediction.
ERGO-II includes other relevant features such as TCRα sequence, V and J genes, MHC and T-cell type (CD4/CD8).
The new ERGO-II configuration is flexible, allowing to include only partial feature information.

## Model Flow
![figure](model_flow.png)

The architecture is flexible to several feature configurations.
TCRβ and peptide sequences are always used, while V and J gene usage is optional.
TCRα sequence usage is optional. Vα and Jα are only used when TCRα is used. MHC usage is optional.
TCRs are encoded with autoencoder or LSTM. Peptides are always encoded with LSTM.
Other features (except for T-Cell type which is not illustrated) are encoded using a learned embedding matrix.
Two MLP are used, one for samples including TCRα and the other for samples missing the TCRα sequence.

## Requirements
ERGO-II is built using [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning), a PyTorch wrapper.

## Instructions
Instructions for training and predicting will be available soon.
Meanwhile you explore [ERGO webtool](http://tcr.cs.biu.ac.il/),
and [ERGO manuscript](https://www.frontiersin.org/articles/10.3389/fimmu.2020.01803/full)
for an explanation about our suggested evaluation methods. 