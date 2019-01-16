# this script is implemented to showcase simple interfaces to our RL leanring pipeline
# two functions are tested in this script:
# fc, att = captioner.compute_img_feat_batch(images)
# fc: N x 2048, att: N x 14 x 14 x2048
# and
# seq_id, sentences = captioner.gen_caption_from_feat(feat_tuple_target, feat_tuple_target)
# seq_id: N x 16, sentences: N string sentences

import torch
import math
from captioner import captioner
import random
import numpy as np
from sys import platform
import pickle
from torch.autograd import Variable
macOS = (platform == 'darwin')
use_cuda = torch.cuda.is_available() and not macOS

class SynUser:
    def __init__(self):
        # load trained model
        params = {}
        params['model'] = 'resnet101'
        params['model_root'] = 'captioner/neuraltalk2'
        params['att_size'] = 7
        model_dir = 'caption_models'
        self.captioner_relative = captioner.Captioner(is_relative= True, model_path= model_dir, image_feat_params= params)
        self.captioner_relative.opt['use_att'] = True
        # build voc
        self.vocabSize = self.captioner_relative.get_vocab_size()

        # load pre-computed data rep.
        fc = np.load('features/fc_feature.npz')['arr_0']
        att = np.load('features/att_feature.npz')['arr_0']
        fc = torch.FloatTensor(fc)
        att = torch.FloatTensor(att)
        print('Data loading completed')
        print('fc.size', fc.size())
        print('att.size', att.size())
        N = att.size(0)
        random.seed(0)
        idx = list(range(N))
        random.shuffle(idx)
        idx = torch.LongTensor(idx)
        # first 10K training
        split = 10000
        self.train_idx = idx[0:split]
        self.test_idx = idx[split::]

        # train
        self.train_fc_input = fc[0:split]
        self.train_att_input = att[0:split]
        # test
        self.test_fc_input = fc[split::]
        self.test_att_input = att[split::]

        absolute_feature = pickle.load(open('features/256embedding.p', 'rb'))
        self.train_feature = torch.FloatTensor(absolute_feature['train'])
        self.test_feature = torch.FloatTensor(absolute_feature['test'])

        self.train_index = torch.arange(0, self.train_fc_input.size(0)).long()
        self.test_index = torch.arange(0, self.test_fc_input.size(0)).long()

        print('init. done!\n#img: {} / {}'.format(self.train_fc_input.size(0), self.test_fc_input.size(0)))
        print('use cuda:', use_cuda)

        return

    def sample_idx(self, img_idx, train_mode):
        if train_mode:
            input = self.train_fc_input
        else:
            input = self.test_fc_input

        for i in range(img_idx.size(0)):
            img_idx[i] = random.randint(0, input.size(0) - 1)
        return

    def get_feedback(self, act_idx, user_idx, train_mode=True):
        if train_mode:
            fc = self.train_fc_input
            att = self.train_att_input
        else:
            fc = self.test_fc_input
            att = self.test_att_input

        act_fc = fc[act_idx]
        act_att = att[act_idx]
        user_fc = fc[user_idx]
        user_att = att[user_idx]
        if use_cuda:
            act_fc = act_fc.cuda()
            act_att = act_att.cuda()
            user_fc = user_fc.cuda()
            user_att = user_att.cuda()
        seq_label, sents_label = self.captioner_relative.gen_caption_from_feat((Variable(user_fc, volatile=True),
                                                                                Variable(user_att, volatile=True)),
                                                                               (Variable(act_fc, volatile=True),
                                                                                Variable(act_att, volatile=True)))

        res = torch.LongTensor(seq_label.size(0), 16).zero_()
        len = seq_label.size(1)
        res[:, 0:len].copy_(seq_label)
        return res

    def get_feedback_with_sent(self, act_idx, user_idx, train_mode=True):
        if train_mode:
            fc = self.train_fc_input
            att = self.train_att_input
        else:
            fc = self.test_fc_input
            att = self.test_att_input

        act_fc = fc[act_idx]
        act_att = att[act_idx]
        user_fc = fc[user_idx]
        user_att = att[user_idx]
        if use_cuda:
            act_fc = act_fc.cuda()
            act_att = act_att.cuda()
            user_fc = user_fc.cuda()
            user_att = user_att.cuda()
        seq_label, sents_label = self.captioner_relative.gen_caption_from_feat((Variable(user_fc, volatile=True),
                                                                                Variable(user_att, volatile=True)),
                                                                               (Variable(act_fc, volatile=True),
                                                                                Variable(act_att, volatile=True)))

        res = torch.LongTensor(seq_label.size(0), 16).zero_()
        len = seq_label.size(1)
        res[:, 0:len].copy_(seq_label)
        return res, sents_label
