# Relative Captioning Dataset
`relative_captions_shoes.json` contains relative expressions which describe fine-grained visual differences on 10,751 pairs of shoe images. The data is in the following format: 

```
{
   "ImageName": "img_womens_clogs_851.jpg", 
   "ReferenceImageName": "img_womens_clogs_512.jpg", 
   "RelativeCaption": "is more of a textured material"
},
```
To obtain the image files, please download the zip file from [Attribute Discovery Dataset](http://tamaraberg.com/attributesDataset/index.html). After unzipping the folder, you could find the images by their names inside the  `womens_*` folders.

## Simple statistics of the dataset 

The following figure shows the length distribution of the relative captions and a few examples from the dataset. 
![](https://github.ibm.com/Xiaoxiao-Guo/interactive_image_retrieval/blob/master/dataset/misc/simple_stats.jpeg?raw=true "")

Most relative expressions contain composite phrases on more than one types of visual feature. And a signifant portion of the data contains propositional phrases that provide information about spatial or structural details.
![](https://raw.github.ibm.com/Xiaoxiao-Guo/interactive_image_retrieval/master/dataset/misc/example_captions.jpeg?token=AAA3CsLwAyNO4iiBZX9hL5zic-pMtPetks5cJPXWwA%3D%3D "")

We tested a few simple baseline methods for the task of relative image captioning. Using the feature concatenation of the two images as the input (ResNet101 pre-trained on ImageNet), the [Show and Tell](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf) based model resulted `26.3` on BLEU-1 and the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) based model produced `29.6` on BLEU-1.

Please refer to the [supplemental material](https://arxiv.org/pdf/1805.00145.pdf) of the paper to see more details on the annotation interface (Supp. A), dataset visualization (Supp. B), baseline performance on the relative image captioner (Supp. C).

## Augment the training data for the user simulator
In our experiment, we found that when the target image and the reference image are visually distinct, users often rely on the visual apprearance of the target image directly and the provided relative descriptions are similar to single-image captions. So, we augmented the size of the dataset to train the user simulator by leveraging additional single-image captioning annotations.  `captions_shoes.json` contains captions on `3,600` shoe images. For the experiments in the paper, we paired each image in this set with five visually distinct reference images. The user simulator was trained using both the relative captions from and this argumented set.

## References 

If find this dataset useful, please cite the following paper:

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
* Please cite the [Attribute Discovery](http://tamaraberg.com/attributesDataset/index.html) paper if you use the original image files. 
* Please cite the [WhittleSearch](http://vision.cs.utexas.edu/whittlesearch/) paper if you use relative attribute labels in your experiment.



