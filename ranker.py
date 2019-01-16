from __future__ import print_function
import torch
from torch.autograd import Variable
import math

class Ranker():
    def __init__(self):
        super(Ranker, self).__init__()
        return

    def update_rep(self, model, input, batch_size=64):
        self.feat = torch.Tensor(input.size(0), model.rep_dim)

        if torch.cuda.is_available():
            self.feat = self.feat.cuda()

        for i in range(1, math.ceil(input.size(0) / batch_size)):
            x = input[(i-1)*batch_size:(i*batch_size)]
            if torch.cuda.is_available():
                x = x.cuda()

            x = Variable(x)
            out = model.forward_image(x)
            self.feat[(i-1)*batch_size:i*batch_size].copy_(out.data)

        if input.size(0) % batch_size > 0:
            x = input[-(input.size(0) % batch_size)::]
            if torch.cuda.is_available():
                x = x.cuda()
            x = Variable(x)
            out = model.forward_image(x)
            self.feat[-(input.size(0) % batch_size)::].copy_(out.data)
        # print(self.feat)
        return

    def compute_rank(self, input, target_idx):
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]

        value = target - input
        value = value ** 2
        value = value.sum(1)
        rank = torch.LongTensor(value.size(0))
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            rank[i] = val.lt(value[i]).sum()

        return rank

    def nearest_neighbor(self, target):
        # L2 case
        idx = torch.LongTensor(target.size(0))
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = val.min(0)
            idx[i] = id[0]
        return idx

    def k_nearest_neighbors(self, target, K = 10):
        idx = torch.LongTensor(target.size(0), K)
        if torch.cuda.is_available():
            target = target.cuda()
            self.feat = self.feat.cuda()

        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = torch.topk(val, k=K, dim=0, largest=False)
            idx[i].copy_(id.view(-1))
        return idx