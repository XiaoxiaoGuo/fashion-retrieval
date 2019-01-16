from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sim_user
import math
import ranker
import random
import time
import sys
from model import NetSynUser


class TripletLossIP(nn.Module):
    def __init__(self, margin):
        super(TripletLossIP, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, average=True):
        dist = torch.sum(
                (anchor - positive) ** 2 - (anchor - negative) ** 2 ,
                dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)
        if average:
            return torch.mean(dist_hinge)
        else:
            return dist_hinge


parser = argparse.ArgumentParser(description='Interactive Image Retrieval')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model-folder', type=str, default="models/",
                    help='triplet loss margin ')
# learning
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                    help='triplet loss margin ')
# exp. control
parser.add_argument('--train-turns', type=int, default=5,
                    help='dialog turns for training')
parser.add_argument('--test-turns', type=int, default=5,
                    help='dialog turns for testing')
args = parser.parse_args()

user = sim_user.SynUser()
ranker = ranker.Ranker()

model = NetSynUser(user.vocabSize + 1)
triplet_loss = TripletLossIP(margin=args.triplet_margin)

if torch.cuda.is_available():
    model.cuda()
    triplet_loss.cuda()


# experiment monitor
class ExpMonitor:
    def __init__(self, train_mode):
        self.train_mode = train_mode
        if train_mode:
            num_turns = args.train_turns
            num_act = user.train_fc_input.size(0)
        else:
            num_turns = args.test_turns
            num_act = user.test_fc_input.size(0)
        self.loss = torch.Tensor(num_turns).zero_()
        self.all_loss = torch.Tensor(num_turns).zero_()
        self.rank = torch.Tensor(num_turns).zero_()
        self.all_rank = torch.Tensor(num_turns).zero_()
        self.count = 0.0
        self.all_count = 0.0
        self.start_time = time.time()
        self.pos_idx = torch.Tensor(num_act).zero_()
        self.neg_idx = torch.Tensor(num_act).zero_()
        self.act_idx = torch.Tensor(num_act).zero_()
        return

    def log_step(self, ranking, loss, user_img_idx, neg_img_idx, act_img_idx, k):
        tmp_rank = ranking.float().mean()
        self.rank[k] += tmp_rank
        self.all_rank[k] += tmp_rank
        self.loss[k] += loss[0]
        self.all_loss[k] += loss[0]
        for i in range(user_img_idx.size(0)):
            self.pos_idx[user_img_idx[i]] += 1
            self.neg_idx[neg_img_idx[i]] += 1
            self.act_idx[act_img_idx[i]] += 1
        self.count += 1
        self.all_count += 1
        return

    def print_interval(self, epoch, batch_idx, num_epoch):
        if self.train_mode:
            output_string = 'Train Epoch:'
            num_input = user.train_fc_input.size(0)
        else:
            output_string = 'Eval Epoch:'
            num_input = user.test_fc_input.size(0)

        output_string += '{} [{}/{} ({:.0f}%)]\tTime:{:.2f}\tNumAct:{}\n'.format(
            epoch, batch_idx, num_epoch, 100. * batch_idx / num_epoch, time.time() - self.start_time, self.pos_idx.sum()
        )
        output_string += 'pos:({:.0f}, {:.0f}) \tneg:({:.0f}, {:.0f}) \tact:({:.0f}, {:.0f})\n'.format(
            self.pos_idx.max(), self.pos_idx.min(), self.neg_idx.max(), self.neg_idx.min(), self.act_idx.max(), self.act_idx.min()
        )

        if self.train_mode:
            dialog_turns = args.train_turns
        else:
            dialog_turns = args.test_turns
        self.rank.mul_(dialog_turns / self.count)
        self.loss.mul_(1.0 / self.count)
        output_string += 'rank:'
        for i in range(dialog_turns):
            output_string += '{:.4f}\t '.format(self.rank[i] / num_input)
        output_string += '\nloss:'
        for i in range(dialog_turns):
            output_string += '{:.4f}\t '.format(self.loss[i])
        print(output_string)
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()
        return

    def print_all(self, epoch):
        if self.train_mode:
            num_input = user.train_fc_input.size(0)
        else:
            num_input = user.test_fc_input.size(0)

        if self.train_mode:
            dialog_turns = args.train_turns
        else:
            dialog_turns = args.test_turns
        self.all_rank.mul_(dialog_turns / self.all_count)
        self.all_loss.mul_(1.0 / self.all_count)
        output_string = '{} #rank:'.format(epoch)
        for i in range(dialog_turns):
            output_string += '{:.4f}\t '.format(self.all_rank[i] / num_input)
        output_string += '\n{} #loss:'.format(epoch)
        for i in range(dialog_turns):
            output_string += '{:.4f}\t '.format(self.all_loss[i])
        print(output_string)

        self.all_loss.zero_()
        self.all_rank.zero_()
        self.all_count = 0.0
        self.loss.zero_()
        self.rank.zero_()
        self.count = 0.0
        sys.stdout.flush()
        return


random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def train_sl(epoch, optimizer):
    print('train epoch #{}'.format(epoch))
    model.train()
    triplet_loss.train()
    exp_monitor_candidate = ExpMonitor(train_mode=True)
    # train / test
    all_input = user.train_feature
    dialog_turns = args.train_turns

    user_img_idx = torch.LongTensor(args.batch_size)
    act_img_idx = torch.LongTensor(args.batch_size)
    neg_img_idx = torch.LongTensor(args.batch_size)
    num_epoch = math.ceil(all_input.size(0) / args.batch_size)

    for batch_idx in range(1, num_epoch + 1):
        # sample target images and first turn feedback images
        user.sample_idx(user_img_idx, train_mode=True)
        user.sample_idx(act_img_idx, train_mode=True)

        ranker.update_rep(model, all_input)
        model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()
        outs = []

        act_input = all_input[act_img_idx]
        if torch.cuda.is_available():
            act_input = act_input.cuda()
        act_input = Variable(act_input)

        act_emb = model.forward_image(act_input)

        for k in range(dialog_turns):
            # get relative captions from user model given user target images and feedback images
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx, train_mode=True)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()
            txt_input = Variable(txt_input)
            # update the query action vector given feedback image and text feedback in this turn
            action = model.merge_forward(act_emb, txt_input)
            # obtain the next turn's feedback images
            act_img_idx = ranker.nearest_neighbor(action.data)

            # sample negative images for triplet loss
            user.sample_idx(neg_img_idx, train_mode=True)

            user_input = all_input[user_img_idx]
            neg_input = all_input[neg_img_idx]
            new_act_input = all_input[act_img_idx]
            if torch.cuda.is_available():
                user_input = user_input.cuda()
                neg_input = neg_input.cuda()
                new_act_input = new_act_input.cuda()
            user_input, neg_input, new_act_input = Variable(user_input), Variable(neg_input), Variable(new_act_input)

            new_act_emb = model.forward_image(new_act_input)
            # ranking and loss
            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            user_emb = model.forward_image(user_input)
            neg_emb = model.forward_image(neg_input)
            loss = triplet_loss.forward(action, user_emb, neg_emb)

            outs.append(loss)
            act_emb = new_act_emb
            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.data, user_img_idx, neg_img_idx, act_img_idx, k)

        # finish dialog and update model parameters
        optimizer.zero_grad()
        outs = torch.stack(outs, dim=0).mean()
        outs.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_epoch)
    exp_monitor_candidate.print_all(epoch)
    return


def eval(epoch):
    print('eval epoch #{}'.format(epoch))
    model.eval()
    triplet_loss.eval()
    exp_monitor_candidate = ExpMonitor(train_mode=False)
    # train / test
    all_input = user.test_feature
    dialog_turns = args.test_turns

    user_img_idx = torch.LongTensor(args.batch_size)
    act_img_idx = torch.LongTensor(args.batch_size)
    neg_img_idx = torch.LongTensor(args.batch_size)
    num_epoch = math.ceil(all_input.size(0) / args.batch_size)
    ranker.update_rep(model, all_input)
    for batch_idx in range(1, num_epoch + 1):
        # sample data index
        user.sample_idx(user_img_idx,  train_mode=False)
        user.sample_idx(act_img_idx, train_mode=False)

        model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            model.hx = model.hx.cuda()

        outs = []

        act_input = all_input[act_img_idx]
        if torch.cuda.is_available():
            act_input = act_input.cuda()
        act_input = Variable(act_input, volatile=True)
        act_emb = model.forward_image(act_input)

        for k in range(dialog_turns):
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx, train_mode=False)
            user.sample_idx(neg_img_idx, train_mode=False)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()
            txt_input = Variable(txt_input, volatile=True)

            action = model.merge_forward(act_emb, txt_input)
            act_img_idx = ranker.nearest_neighbor(action.data)
            user_input = all_input[user_img_idx]
            neg_input = all_input[neg_img_idx]
            new_act_input = all_input[act_img_idx]
            if torch.cuda.is_available():
                user_input = user_input.cuda()
                neg_input = neg_input.cuda()
                new_act_input = new_act_input.cuda()
            user_input, neg_input, new_act_input = Variable(user_input, volatile=True), Variable(neg_input, volatile=True), Variable(new_act_input, volatile=True)
            new_act_emb = model.forward_image(new_act_input)

            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            user_emb = model.forward_image(user_input)
            neg_emb = model.forward_image(neg_input)
            loss = triplet_loss.forward(action, user_emb, neg_emb)

            outs.append(loss)
            act_emb = new_act_emb

            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.data, user_img_idx, neg_img_idx, act_img_idx, k)

        if batch_idx % args.log_interval == 0:
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_epoch)
    exp_monitor_candidate.print_all(epoch)
    return

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=1e-8)
for epoch in range(1, args.epochs + 1):
    train_sl(epoch, optimizer)
    eval(epoch)
    torch.save(model.state_dict(), (args.model_folder+'sl-{}.pt').format(epoch))