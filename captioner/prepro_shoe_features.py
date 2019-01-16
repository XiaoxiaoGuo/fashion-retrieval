from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from neuraltalk2.misc.resnet_utils import myResnet
import neuraltalk2.misc.resnet as resnet

def compute_img_feat(img_name, im_path, my_resnet):
    # load the image
    I = skimage.io.imread(os.path.join(im_path, img_name))

    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)

    I = I.astype('float32') / 255.0
    I = torch.from_numpy(I.transpose([2, 0, 1]))
    if torch.cuda.is_available(): I = I.cuda()
    I = Variable(preprocess(I), volatile=True)
    fc, att = my_resnet(I, params['att_size'])

    return fc.data.cpu().float().numpy(), att.data.cpu().float().numpy()

def make_dir_if_not_there(d):
    if not os.path.isdir(d): os.mkdir(d)

def main(args):

    if args.is_relative:
        featureDirPrefix = os.path.join(args.output_dir, 'features_relative')
    else:
        featureDirPrefix = os.path.join(args.output_dir, 'features_direct')

    dir_fc = featureDirPrefix + '_fc'
    dir_att = featureDirPrefix + '_att'

    make_dir_if_not_there(args.output_dir)
    make_dir_if_not_there(dir_fc)
    make_dir_if_not_there(dir_att)

    imageDir = args.image_dir

    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
    my_resnet = myResnet(net)
    if torch.cuda.is_available():
        print('cuda available, use cuda')
        my_resnet.cuda()
    my_resnet.eval()

    imgs_with_captions = json.load(open(args.json_file, 'r'))
    N = len(imgs_with_captions)

    seed(123) # make reproducible

    idx_start = int(N*args.start)
    idx_end = int(N*args.end)

    print('start', idx_start, 'end', idx_end)

    for i in range(idx_start, idx_end):
        im_A_id = imgs_with_captions[i]['image_id'][0][4:-4]
        im_B_id = imgs_with_captions[i]['image_id'][1][4:-4]
        curr_id = im_A_id + '_' + im_B_id

        curr_images = imgs_with_captions[i]['image_id']

        imName = curr_images[0]
        tmp_fc, tmp_att = compute_img_feat(imName, imageDir, my_resnet)

        if args.is_relative:

            imName_ref = curr_images[1]
            tmp_fc_ref,  tmp_att_ref= compute_img_feat(imName_ref, imageDir, my_resnet)
            tmp_fc = np.concatenate((tmp_fc, tmp_fc_ref), axis= -1)
            tmp_att = np.concatenate((tmp_att, tmp_att_ref), axis= -1)

        np.save(os.path.join(dir_fc, curr_id), tmp_fc)
        np.savez_compressed(os.path.join(dir_att, curr_id), feat=tmp_att)

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
            sys.stdout.flush()

    print('Feature preprocessing done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--is_relative', type= str, default= 'True')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='neuraltalk2', type=str, help='model root')
    parser.add_argument('--output_dir', default='debug_output', type=str, help='temp output folder')
    parser.add_argument('--image_dir', default='/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k', type=str)
    parser.add_argument('--json_file', default='relativeCaptionAMTdata/relative_caption_batch123_denoised.json', type=str, help='')
    parser.add_argument('--start', default= 0.1, type= float)
    parser.add_argument('--end', default= 0.2, type= float)

    args = parser.parse_args()
    args.is_relative = args.is_relative == 'True'

    print('parsed input parameters:')
    params = vars(args)  # convert to ordinary dict
    print(json.dumps(params, indent = 2))

    main(args)