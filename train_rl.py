from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import ranker
import random
import time
import sys
import sim_user
from model import NetSynUser
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=128,
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=15,
                    help='number of epochs to train')
# learning
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--tau', type=float, default=1,
                    help='softmax temperature')
parser.add_argument('--seed', type=int, default=7771,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--neg-num', type=int, default=5,
                    help='number of negative candidates in the denominator')
parser.add_argument('--model-folder', type=str, default="models/",
                    help='triplet loss margin ')

parser.add_argument('--top-k', type=int, default=4,
                    help='top k candidate for policy and nearest neighbors')
parser.add_argument('--pretrained-model', type=str, default="models/sl-12.pt",
                    help='path to pretrained sl model')
parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                    help='triplet loss margin ')
# exp. control
parser.add_argument('--train-turns', type=int, default=5,
                    help='dialog turns for training')
parser.add_argument('--test-turns', type=int, default=5,
                    help='dialog turns for testing')


args = parser.parse_args()


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


# experiment monitor
class ExpMonitor():
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
        self.act_idx = torch.Tensor(num_act).zero_()
        return

    def log_step(self, ranking, loss, user_img_idx, act_img_idx, k):
        tmp_rank = ranking.float().mean()
        self.rank[k] += tmp_rank
        self.all_rank[k] += tmp_rank
        self.loss[k] += loss[0]
        self.all_loss[k] += loss[0]
        for i in range(user_img_idx.size(0)):
            self.pos_idx[user_img_idx[i]] += 1
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
        output_string += 'pos:({:.0f}, {:.0f}) \tact:({:.0f}, {:.0f})\n'.format(
            self.pos_idx.max(), self.pos_idx.min(), self.act_idx.max(), self.act_idx.min()
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


user = sim_user.SynUser()
ranker = ranker.Ranker()

behavior_model = NetSynUser(user.vocabSize + 1)
target_model = NetSynUser(user.vocabSize + 1)
triplet_loss = TripletLossIP(margin=args.triplet_margin)
# load pre-trained model
behavior_model.load_state_dict(torch.load(args.pretrained_model, map_location=lambda storage, loc: storage))
# load pre-trained model
target_model.load_state_dict(torch.load(args.pretrained_model, map_location=lambda storage, loc: storage))

if torch.cuda.is_available():
    behavior_model.cuda()
    target_model.cuda()
    triplet_loss.cuda()


def rollout_search(behavior_state, target_state, cur_turn, max_turn, user_img_idx, all_input):
    # 1. compute the top-k nearest neighbor for current state
    top_k_act_img_idx = ranker.k_nearest_neighbors(target_state.data, K=args.top_k)

    # 2. rollout for each candidate in top k
    target_hx_bk = target_model.hx
    rollout_values = []
    for i in range(args.top_k):
        target_model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            target_model.hx = target_model.hx.cuda()
        target_model.hx.data.copy_(target_hx_bk.data)
        act_img_idx = top_k_act_img_idx[:, i]
        score = 0
        for j in range(max_turn - cur_turn):
            txt_input = user.get_feedback(act_idx=act_img_idx, user_idx=user_img_idx,  train_mode=True)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()
            txt_input = Variable(txt_input, volatile=True)

            if torch.cuda.is_available():
                act_img_idx = act_img_idx.cuda()
            act_emb = ranker.feat[act_img_idx]

            action = target_model.merge_forward(Variable(act_emb, volatile=True), txt_input)
            act_img_idx = ranker.nearest_neighbor(action.data)
            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            score = score + ranking_candidate
        rollout_values.append(score)
    rollout_values = torch.stack(rollout_values, dim=1)
    # compute greedy actions
    _, greedy_idx = rollout_values.min(dim=1)
    # recover target_state
    target_model.hx = target_hx_bk
    if torch.cuda.is_available():
        greedy_idx = greedy_idx.cuda()

    act_opt = torch.gather(top_k_act_img_idx, 1, greedy_idx.cpu().unsqueeze(1)).view(-1)

    # 3. compute loss
    # compute the log prob for candidates
    dist_action = []
    act_input = all_input[act_opt]
    if torch.cuda.is_available():
        act_input = act_input.cuda()
    act_emb = behavior_model.forward_image(Variable(act_input))
    dist = -torch.sum((behavior_state - act_emb) ** 2, dim=1) / args.tau
    dist_action.append(dist)
    for i in range(args.neg_num):
        neg_img_idx = torch.LongTensor(args.batch_size)
        user.sample_idx(neg_img_idx, train_mode=True)

        neg_input = all_input[neg_img_idx]
        if torch.cuda.is_available():
            neg_input = neg_input.cuda()
        neg_emb = behavior_model.forward_image(Variable(neg_input))
        dist = -torch.sum((behavior_state - neg_emb) ** 2, dim=1) / args.tau
        dist_action.append(dist)
    dist_action = torch.stack(dist_action, dim=1)
    label_idx = torch.LongTensor(args.batch_size).fill_(0)
    if torch.cuda.is_available():
        label_idx = label_idx.cuda()
    loss = torch.nn.functional.cross_entropy(input=dist_action, target=Variable(label_idx))
    # compute the reg following the pre-training loss
    if torch.cuda.is_available():
        user_img_idx = user_img_idx.cuda()
    target_emb = ranker.feat[user_img_idx]
    reg = torch.sum((behavior_state - Variable(target_emb)) ** 2, dim=1).mean()

    return act_opt, reg + loss


user_img_idx_ = torch.LongTensor(args.batch_size)
act_img_idx_ = torch.LongTensor(args.batch_size)
user.sample_idx(user_img_idx_, train_mode=True)
user.sample_idx(act_img_idx_, train_mode=True)
def train_rl(epoch, optimizer):
    behavior_model.set_rl_mode()
    target_model.eval()
    triplet_loss.train()
    exp_monitor_candidate = ExpMonitor(train_mode=True)
    # train / test
    all_input = user.train_feature
    dialog_turns = args.train_turns
    #
    user_img_idx = torch.LongTensor(args.batch_size)
    act_img_idx = torch.LongTensor(args.batch_size)

    # update ranker
    ranker.update_rep(target_model, all_input)
    num_epoch = math.ceil(all_input.size(0) / args.batch_size)

    for batch_idx in range(1, num_epoch + 1):
        # sample data index
        user.sample_idx(user_img_idx, train_mode=True)
        user.sample_idx(act_img_idx, train_mode=True)

        target_model.init_hid(args.batch_size)
        behavior_model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            target_model.hx = target_model.hx.cuda()
            behavior_model.hx = behavior_model.hx.cuda()

        loss_sum = 0
        for k in range(dialog_turns):
            # construct data
            txt_input = user.get_feedback(act_idx=act_img_idx.cpu(), user_idx=user_img_idx.cpu(), train_mode=True)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()

            # update model part
            if torch.cuda.is_available():
                act_img_idx = act_img_idx.cuda()
            act_emb = ranker.feat[act_img_idx]
            behavior_state = behavior_model.merge_forward(Variable(act_emb), Variable(txt_input))

            # update base model part
            target_state = target_model.merge_forward(Variable(act_emb, volatile=True),
                                                      Variable(txt_input, volatile=True))

            ranking_candidate = ranker.compute_rank(behavior_state.data, user_img_idx)

            act_img_idx_mc, loss = rollout_search(behavior_state, target_state, k, dialog_turns, user_img_idx, all_input)

            loss_sum = loss + loss_sum

            act_img_idx.copy_(act_img_idx_mc)

            exp_monitor_candidate.log_step(ranking_candidate, loss.data, user_img_idx, act_img_idx, k)

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('# candidate ranking #')
            exp_monitor_candidate.print_interval(epoch, batch_idx, num_epoch)

    print('# candidate ranking #')
    exp_monitor_candidate.print_all(epoch)
    return

def eval(epoch):
    # train_mode = True
    print('eval epoch #{}'.format(epoch))
    behavior_model.eval()
    triplet_loss.eval()
    train_mode = False
    all_input = user.test_feature
    dialog_turns = args.test_turns

    exp_monitor_candidate = ExpMonitor(train_mode=train_mode)

    user_img_idx = torch.LongTensor(args.batch_size)
    act_img_idx = torch.LongTensor(args.batch_size)
    neg_img_idx = torch.LongTensor(args.batch_size)
    num_epoch = math.ceil(all_input.size(0) / args.batch_size)


    ranker.update_rep(behavior_model, all_input)
    for batch_idx in range(1, num_epoch + 1):
        # sample data index

        user.sample_idx(user_img_idx,  train_mode=train_mode)
        user.sample_idx(act_img_idx, train_mode=train_mode)

        behavior_model.init_hid(args.batch_size)
        if torch.cuda.is_available():
            behavior_model.hx = behavior_model.hx.cuda()

        if torch.cuda.is_available():
            act_img_idx = act_img_idx.cuda()
        act_emb = ranker.feat[act_img_idx]

        for k in range(dialog_turns):
            txt_input = user.get_feedback(act_idx=act_img_idx.cpu(), user_idx=user_img_idx.cpu(), train_mode=train_mode)
            if torch.cuda.is_available():
                txt_input = txt_input.cuda()
            txt_input = Variable(txt_input, volatile=True)

            action = behavior_model.merge_forward(Variable(act_emb, volatile=True), txt_input)
            act_img_idx = ranker.nearest_neighbor(action.data)

            user.sample_idx(neg_img_idx, train_mode=train_mode)
            if torch.cuda.is_available():
                user_img_idx = user_img_idx.cuda()
                neg_img_idx = neg_img_idx.cuda()
                act_img_idx = act_img_idx.cuda()

            user_emb = ranker.feat[user_img_idx]
            neg_emb = ranker.feat[neg_img_idx]

            new_act_emb = ranker.feat[act_img_idx]

            ranking_candidate = ranker.compute_rank(action.data, user_img_idx)
            loss = triplet_loss.forward(action, Variable(user_emb), Variable(neg_emb))
            act_emb = new_act_emb

            # log
            exp_monitor_candidate.log_step(ranking_candidate, loss.data, user_img_idx,  act_img_idx, k)
    exp_monitor_candidate.print_all(epoch)
    return


optimizer = optim.Adam(behavior_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
for i in range(20):
    eval(i)
    train_rl(i, optimizer)
    torch.save(behavior_model.state_dict(), (args.model_folder+'rl-{}.pt').format(i))

eval(20)
