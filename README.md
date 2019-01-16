# Dialog-based interactive image retrieval 

## About this repository
This repository contains an implementation of the models introduced in the paper [Dialog-based Interactive Image Retrieval](https://papers.nips.cc/paper/7348-dialog-based-interactive-image-retrieval.pdf). The model taks an image and a graph proposal as input and predicts the object and relationship categories in the graph. The network  is implemented using [PyTorch](https://pytorch.org/) and the rest of the framework is in Python. The user model is built directly on top of [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). 

## Citing this work
If you find this work useful in your research, please consider citing:
```
@incollection{NIPS2018_7348,
title = {Dialog-based Interactive Image Retrieval},
author = {Guo, Xiaoxiao and Wu, Hui and Cheng, Yu and Rennie, Steven and Tesauro, Gerald and Feris, Rogerio},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {676--686},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7348-dialog-based-interactive-image-retrieval.pdf}
}
```
## Project page
The project page is available at [https://www.spacewu.com/posts/fashion-retrieval/](https://www.spacewu.com/posts/fashion-retrieval/).

## Dependencies
To get started with the framework, install the following dependencies:
- Python 3.6
- [PyTorch 0.3](https://pytorch.org/get-started/previous-versions/)

## Dataset
The  dataset used in the paper is built on the [Attribute Discovery Dataset](http://tamaraberg.com/attributesDataset/index.html). Please refer to the [dataset README](dataset/) for our dataset details. The pre-computed image features and user captioning model can be downloaded from [here](https://ibm.box.com/s/a1zml3pyx4v8yblvy48oyjt1vsbjbkrk). 


## Train and evaluate a model
Follow the following steps to train a model:
1. Prepare following [dataset instructions](dataset/) or download the pre-computed image features and user captioning model from [here](https://ibm.box.com/s/a1zml3pyx4v8yblvy48oyjt1vsbjbkrk).
2. Move the pre-computed image features into [features folder](features/) and the user captioning model into [caption_models folder](caption_models/). 
3. Edit the training script `experiments/scripts/train.sh` such that all paths agree with the files on your file system.
4. To train the model with the pre-training loss, run:
```
python train_sl.py --log-interval=50 --lr=0.001  --batch-size=128 --model-folder="models/"
```
The program saves trained models into the folder `models/` every epoch. 

5. To fine-tune the final model with the policy improvement loss, run:
```
python train_rl.py --log-interval=10 --lr=0.0001 --top-k=4 --batch-size=128 --tau=0.2 --pretrained-model="models/sl-10.pt" --save-folder="models/"
```
The program saves trained models into the folder `models/` every epoch. 


## License
MIT License
