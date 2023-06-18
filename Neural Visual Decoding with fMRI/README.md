# Neural Visual Decoding with fMRI

**Final Project - “Neural Networks” Course at Sapienza University of Rome, A.Y. 2022/2023** \
Professor Danilo Comminiello, Professor Simone Scardapane <br>
Author: Salvatore Falciglia, Master Student in Artificial Intelligence and Robotics - Sapienza, University of Rome

## 📜Short Abstract
Visual neural decoding, or the ability to interpret external visual stimuli from patterns of brain activity, is a challenging task in neuroscience research. Recent advances in generative deep learning methods have made it possible to decode brain activity patterns and reconstruct visual stimuli from the neural response they evoke. In this work, I propose a re-implementation from scratch of the architecture presented in [[1]](https://www.sciencedirect.com/science/article/pii/S1053811920310879), dealing with the training process of the **VAE-GAN model**, which the implemented architecture is based on. <br>
**The results and plots of the training and validation losses are all shown in the .ipynb notebook in the repository, along with all the details necessary to understand the model implementation.**

### Dataset
This dataset is referred to as 69 dataset [[2]](https://pubmed.ncbi.nlm.nih.gov/20858128/), which contains one participant presented a hundred of handwritten gray-scale digits (6 s and 9 s) with the size of 28×28. Each character remained visible for 12.5 s and flickered at a rate of 6 Hz, with fixation, in a 3T scanner (TR, 2.5 s; voxel size, 2×2×2 mm). The images are taken from the training set of the MNIST and the fMRI data are taken from V1, V2, and V3. Following the original train/test set split, ninety stimulusfMRI paired examples are used for training and the rest for validation. <br>
The Dataset can be downloaded from [here](https://data.donders.ru.nl/collections/di/dcc/DSC_2018.00112_485?0), by requesting access. [For the sake of semplicity, only for the university project presented here, the dataset is already in the repository]


## 👨‍💻👩‍💻How to run it
1) Clone the repository on your system. <br>
2) Be sure to have PyTorch Lightning installed on your system, otherwise install it: <br>
<font color="red">`pip install pytorch-lightning`</font> <br>
3) Be sure to change the relative paths within both files <font color="red">`nn_main_69_NEWTRAIN.py`</font> and <font color="red">`nn_utils_69_NEWTRAIN.py`</font>. <br>
4) In order to run the model, just write the following command: <br>
<font color="red">`python3 /<your path>/nn_main_69_NEWTRAIN.py`</font> <br>
  
NOTE: Since it was not possible to upload the .ckpt file with the weights of the model, it wil be necessary to run the model starting from the first epoch. To do this, simply comment out the line where the .ckpt file is recalled in <font color="red">`nn_main_69_NEWTRAIN.py`</font>: <br>
  
```ruby
# Training and Evaluating the NeuralVisualDecodingfMRIModel Model
trainer_kay.fit(model,
                visMNIST_data_module.train_dataloader(),
                visMNIST_data_module.val_dataloader()),
                #ckpt_path='/raid/home/falcigls/NeuralNetworks/DATASET_69/Stuff_NEWTRAIN/TESTS/epoch=1999-step=90000 copy.ckpt')

print("\n\nFIT DONE\n\n")
``` 


## 📝 Further Info:
Pytorch Lightning has been used for this re-implementation. Anyway, since I had to work with multiple optimizers at the same time, the training and optimization processes have been managed manually.
All experiments have been conducted using a V100 GPU with 32 GB of Virtual RAM (VRAM), from INFN, kindly made available by Professor Stefano Giagu. <br>
Moreover, *cuda* and *cuDNN* gave problems to run convolutions due to their versions and compatibility. I tried to disable cuDNN, by setting the *torch.backends.cudnn.enabled* flag to *False*, but this really slower training performances. Eventually, only with one dataset (out of all the datasets exploited in the [**reference paper**](https://www.sciencedirect.com/science/article/pii/S1053811920310879)),  I managed to train efficacely the entire architecture by adjusting the *batch_size* and by choosing proper optimizers. <br>
  
For any doubt or clarification send me an [email](mailto:falciglia.2015426@studenti.uniroma1.it?subject=[GitHub_LTW]).<br>
  
**Further note**: The above material is published here for university project purposes only. All rights reserved.
