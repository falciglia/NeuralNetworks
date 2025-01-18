<div align="center">

# **Learning visual stimulus-evoked EEG manifold <br> for neural image classification**

<p align="center">
<img src="riemanispectranet.png" width="95%">
</p>


[📇 Paper](https://www.sciencedirect.com/science/article/pii/S0925231224004259) |
[🤗 Hugging Face Leaderboard of Models](https://huggingface.co/spaces/DavidVivancos/MindBigData-Leaderboard) |
[📚 MindBigData2022](https://huggingface.co/datasets/DavidVivancos/MindBigData2022) 

[Salvatore Falciglia](https://scholar.google.com/citations?user=E-nObHcAAAAJ&hl=it&oi=ao)<sup>1,3</sup>, 
[Filippo Betello](https://scholar.google.com/citations?user=ZjIo22MAAAAJ&hl=it&oi=ao)<sup>1</sup>,
[Samuele Russo](https://scholar.google.com/citations?user=bvNvDiEAAAAJ&hl=it&oi=ao)<sup>2</sup>, 
[Christian Napoli](https://scholar.google.com/citations?user=NS1JOAUAAAAJ&hl=it)<sup>1,4,5#</sup>

<sup>1</sup>Department of Computer, Control and Management Engineering, Via Ariosto 25, Rome, 00185, Italy,   
<sup>2</sup>Department of Psychology, Sapienza University of Rome, Via dei Marsi 78, Rome, 00185, Italy, 
<sup>3</sup>The BioRobotics Institute, Scuola Superiore Sant’Anna, Pisa, 56025, Italy, 
<sup>4</sup>Institute for Systems Analysis and Computer Science, Italian National Research Council, Via dei Taurini 19, Rome, 00185, Italy,
<sup>5</sup>Department of Intelligent Computer Systems, Czestochowa University of Technology, al.Armii Krajowej 36, Czestochowa, 42-200, Poland

<sup>#</sup> Corresponding author

🎧 [▶︎ •၊၊||၊|။|||| | podcast by notebooklm.google | ||||။|၊||၊၊•](https://github.com/falciglia/Sapienza-Rome/raw/refs/heads/main/Neural%20Visual%20Decoding%20with%20EEG/podcast_RieManiSpectraNet.wav) 🎙️

</div>

# Welcome👋🧠

In this paper, we focus on visual Neural Manifold Learning, proposing a novel model, called **RieManiSpectraNet** (**Rie**mann **Mani**fold **Spectr**ogr**a**m **Net**work), constituted by a deep learning architecture for modeling and classifying electroencephalograms that integrates spatio-temporal and spectral features from EEG signals, by revealing the inner latent representation of visually evoked brain signals recorded by EEG. First of all, the system deals separately with spatial information and temporal information by extracting Spatial Covariance Matrices and Feature Maps, respectively. Spatial features are drawn out through Riemann geometry, while Temporal features are brought out through LSTM layers. Both extracted features are then combined to reach a proper embedding to integrate with the spectral features coming from the spectrogram images of each EEG channel computed according to the Short-Time Fourier Transform (STFT) algorithm. In this way, by combining a spatio-temporal- and a spectral-processing stream, we are able to describe the overall activity of the entire brain, delegating the system itself to focus on the correct weighting of features coming from different brain areas.

Future research should explore variations on the proposed architecture, possibly based on attention-based mechanisms. Moving forward, several promising directions for future research emerge from our findings, e.g. in order to explore **Neural Visual Decoding** from non-invasive fMRI imaging techniques, working with natural scenes and natural images, and exploiting deep generative methods to perform **Neural Image Reconstruction**.

**Reading minds, cracking the human brain cortex, is closer than ever before.**

# Highlights🔥

- Introducing a **novel EEG processing strategy**, pioneering the fusion of spatial, temporal, and spectral information.

- Presenting **RieManiSpectraNet** (inspired by [G.Zhang & A.Etemad work](https://ieeexplore.ieee.org/abstract/document/10328680?casa_token=frUzG7BNlFsAAAAA:HaUJRkMLdOttSlhEFSsBhPJcUDlbc8UIj_waeT-0iZ_Rp_lYMElpku3A5PeJxqCJuAhnHUMDhS2O)), harnessing the representational power of the Riemann Manifold and the STFT algorithm.

- Exploring a [novel dataset](https://mindbigdata.com/opendb/visualmnist.html), released on December 27, 2022, fixing the benchmark for it in the context of the Neural Image Classification task.

- Overall accuracy comparable to the [SOTA benchmarks in the field](https://huggingface.co/spaces/DavidVivancos/MindBigData-Leaderboard).


# Reference

- 📄 **Publication July 2024**:
  [Learning visual stimulus-evoked EEG manifold for neural image classification.](https://www.sciencedirect.com/science/article/pii/S0925231224004259)
  S.Falciglia, F.Betello, S.Russo, C.Napoli#. Neurocomputing 2024.
  
```bibtex
@article{falciglia2024learning,
  title={Learning visual stimulus-evoked EEG manifold for neural image classification},
  author={Falciglia, Salvatore and Betello, Filippo and Russo, Samuele and Napoli, Christian},
  journal={Neurocomputing},
  volume={588},
  pages={127654},
  year={2024},
  publisher={Elsevier}
}
```
