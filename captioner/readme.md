The code assumes you can use IBM research cognitive computing clusters, which can access our shoe captioning data for fashion dialog project.


# Download pre-trained resnet 
Download model from [this link](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM)
Place it in "neuraltalk2" folder. Right now it works with ResNet101, haven't tested with other models. 

# Train model 
Raw relative captiong data is on CCC: 

```
# old one: batch 1 relative caption only 
/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/captions/caption_data.json

# please use the new one, which has both batch 1 and batch 2 annotations with manually denoised labels by Hui
/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/captions/relative_caption_batch123_denoised.json
or the same one here:
relativeCaptionAMTdata/relative_caption_batch123_denoised.json

```

## 1 Precompute image ResNet features
```
# Prepare features for relative captioning by concatenating target and reference image features

jbsub -queue x86_6h -cores 1+1 -mem 30g -out logs/preprocess_feature_relative_augmentation.txt \
python prepro_shoe_features.py --output_dir debug_output_augmentation \
--image_dir /dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/imgs_aug \
--json_file relativeCaptionAMTdata/augmentation.json


```
## 2 prepare caption data
```
# Preprocess caption for relative captioning task

python prepro_shoe_labels.py 

```
## 3 training the model
```

jbsub -queue x86_7d -cores 1+1 -mem 30g -out logs/train_relative_captioning_augx4.txt \
python train_shoes_relative.py --is_relative True --output_dir debug_output_augx4


```
`train_shoes_relative.py` will load parameter settings from opts.py. Notice that `opts.py` is exactly the same as `neuraltalk2/opts.py`
excerpt for the following two lines added on .

```
parser.add_argument('--is_relative', type=str, default='True')
parser.add_argument('--output_dir', default='debug_output', type=str, help='temp output folder')
```
# Test the model

## 1 Evaluate the model trained from the last step 
```
# Train captioning for relative captioning task
python eval_shoes_relative.py --is_relative True --output_dir debug_output

```
If you don't want to print lots of text on your screen you can comment the following lines from `neuraltalk2/eval_utils.py`
```
# comment this line
print(cmd)  
os.system(cmd)

# comment these two lines
if verbose:
    print('image %s: %s' %(entry['image_id'], entry['caption']))
```
## 2 Obtain captions on input raw images images 

```
jbsub -queue x86_6h -cores 1+1 -mem 30g -out logs/testCaptioner_forRL.txt \
python testCaptioner_forRL.py 


```
