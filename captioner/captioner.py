from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, math
import numpy as np

import os, sys
from six.moves import cPickle

sys.path.insert(0, 'neuraltalk2')
sys.path.insert(0, 'captioner/neuraltalk2')
sys.path.insert(0, 'captioner/')
print('relative captioning is called')
import models
from dataloader import *
from dataloaderraw import *
import misc.utils as utils
import torch

import skimage.io
from torch.autograd import Variable
from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


from misc.resnet_utils import myResnet
from resnet_batch import ResNetBatch
import misc.resnet as resnet


class object:
    def __init__(self):
        self.input_fc_dir = ''
        self.input_json = ''
        self.batch_size = ''
        self.id = ''
        self.sample_max = 1
        self.cnn_model = 'resnet101'
        self.model = ''
        self.language_eval = 0
        self.beam_size = 2
        self.temperature = 1.0
        return


class Captioner():

    def __init__(self, is_relative= True, model_path= None, image_feat_params= None):
        opt = object()
        # inputs specific to shoe dataset
        infos_path = os.path.join(model_path, 'infos_best.pkl')
        model_name = os.path.join(model_path, 'model_best.pth')

        opt.infos_path = infos_path
        opt.model = model_name
        opt.beam_size = 1
        opt.load_resnet = False

        with open(opt.infos_path, 'rb') as f:
            infos = cPickle.load(f)

        # override and collect parameters
        if len(opt.input_fc_dir) == 0:
            opt.input_fc_dir = infos['opt'].input_fc_dir
            opt.input_att_dir = infos['opt'].input_att_dir
            opt.input_label_h5 = infos['opt'].input_label_h5
        if len(opt.input_json) == 0:
            opt.input_json = infos['opt'].input_json
        if opt.batch_size == 0:
            opt.batch_size = infos['opt'].batch_size
        if len(opt.id) == 0:
            opt.id = infos['opt'].id
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "model"]
        for k in vars(infos['opt']).keys():
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

        vocab = infos['vocab']  # ix -> word mapping

        # Setup the model
        model = models.setup(opt)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.model))
            model.cuda()
        else:
            model.load_state_dict(torch.load(opt.model, map_location={'cuda:0': 'cpu'}))

        model.eval()

        self.is_relative = is_relative
        self.model = model
        self.vocab = vocab
        self.opt = vars(opt)



        if opt.load_resnet:
            net = getattr(resnet, image_feat_params['model'])()
            net.load_state_dict(
                torch.load(os.path.join(image_feat_params['model_root'], image_feat_params['model'] + '.pth')))
            my_resnet = myResnet(net)
            if torch.cuda.is_available():
                my_resnet.cuda()
            my_resnet.eval()

            my_resnet_batch = ResNetBatch(net)
            if torch.cuda.is_available():
                my_resnet_batch.cuda()

            self.my_resnet_batch = my_resnet_batch
            self.my_resnet = my_resnet
        self.att_size = image_feat_params['att_size']

    def gen_caption(self, im_target, im_reference=None):
        if self.is_relative and im_reference == None:
            return ''

        if not self.is_relative and not im_reference == None:
            return ''

        fc_feat, att_feat = self.get_feat(im_target, im_reference)
        tmp = (fc_feat, att_feat)
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feat, att_feat = tmp

        if not self.opt['use_att']:
            att_feat = Variable(torch.zeros(1, 1, 1, 1), volatile=True)

        seq, _ = self.model.sample(fc_feat, att_feat, self.opt)
        sents = utils.decode_sequence(self.vocab, seq)

        return seq, sents


    def gen_caption_from_feat(self, feat_target, feat_reference= None):
        if self.is_relative and feat_reference == None:
            return None, None

        if not self.is_relative and not feat_reference == None:
            return None, None

        if self.is_relative:
            fc_feat = torch.cat((feat_target[0], feat_target[0] - feat_reference[0]), dim= -1)
            att_feat = torch.cat((feat_target[1], feat_target[1] - feat_reference[1]), dim= -1)
        else:
            fc_feat = feat_target[0]
            att_feat = feat_target[1]

        seq, _ = self.model.sample(fc_feat, att_feat, self.opt)
        sents = utils.decode_sequence(self.vocab, seq)

        return seq, sents

    def get_vocab_size(self):
        return len(self.vocab)

    def get_feat(self, im_target, im_referece):

        tmp_fc, tmp_att = self.compute_img_feat_batch(im_target)
        tmp_fc = tmp_fc.data.cpu().float().numpy()
        tmp_att = tmp_att.data.cpu().float().numpy()

        if self.is_relative:
            tmp_fc_ref, tmp_att_ref = self.compute_img_feat_batch(im_referece)
            tmp_fc_ref = tmp_fc_ref.data.cpu().float().numpy()
            tmp_att_ref = tmp_att_ref.data.cpu().float().numpy()

            tmp_fc = np.concatenate((tmp_fc, tmp_fc_ref), axis=-1)
            tmp_att = np.concatenate((tmp_att, tmp_att_ref), axis=-1)

        return tmp_fc, tmp_att

    def get_img_feat(self, img_name):
        # load the image
        I = skimage.io.imread(img_name)

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1]))
        if torch.cuda.is_available(): I = I.cuda()
        I = Variable(preprocess(I), volatile=True)
        fc, att = self.my_resnet(I, self.att_size)

        return fc, att

    def get_img_feat_batch(self, img_names, batchsize= 32):
        if not isinstance(img_names, list):
            img_names = [img_names]

        num_images = len(img_names)
        num_batches = math.ceil(np.float(num_images)/np.float(batchsize))

        feature_fc = []
        feature_att = []

        for id in range(num_batches):
            startInd = id * batchsize
            endInd = min((id+1)*batchsize, num_images)

            img_names_current_batch = img_names[startInd:endInd]
            I_current_batch = []

            for img_name in img_names_current_batch:
                I = skimage.io.imread(img_name)

                if len(I.shape) == 2:
                    I = I[:, :, np.newaxis]
                    I = np.concatenate((I, I, I), axis=2)

                I = I.astype('float32') / 255.0
                I = torch.from_numpy(I.transpose([2, 0, 1]))
                I_current_batch.append(preprocess(I))

            I_current_batch = torch.stack(I_current_batch, dim= 0)
            if torch.cuda.is_available(): I_current_batch = I_current_batch.cuda()
            I_current_batch = Variable(I_current_batch, volatile=True)
            fc, att = self.my_resnet_batch(I_current_batch, self.att_size)

            feature_fc.append(fc)
            feature_att.append(att)

        feature_fc = torch.cat(feature_fc, dim= 0)
        feature_att = torch.cat(feature_att, dim= 0)

        return feature_fc, feature_att



