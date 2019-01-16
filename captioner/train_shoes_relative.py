from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os, sys, json
from six.moves import cPickle

import opts

sys.path.insert(0, 'neuraltalk2')

import neuraltalk2.models as models
from neuraltalk2.dataloader import *
import neuraltalk2.eval_utils as eval_utils
import neuraltalk2.misc.utils as utils

use_cuda = torch.cuda.is_available()
print('Cuda available!')

def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    histories = {}

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    if use_cuda: model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
                sys.stdout.flush()
            update_lr_flag = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')

        if use_cuda: torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        if use_cuda:
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        else:
            tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        if use_cuda: torch.cuda.synchronize()
        end = time.time()

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'test',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
opt.is_relative = opt.is_relative == 'True'

if opt.is_relative:
    opt.fc_feat_size = 4096
    opt.att_feat_size = 4096
    relative_str = 'relative'

else:
    opt.fc_feat_size = 2048
    opt.att_feat_size = 2048
    relative_str = 'direct'

opt.input_json = os.path.join(opt.input_dir, 'caption_{x}.json'.format(x= relative_str))
opt.input_fc_dir = os.path.join(opt.input_dir, 'features_{x}_fc'.format(x= relative_str))
opt.input_att_dir = os.path.join(opt.input_dir, 'features_{x}_att'.format(x= relative_str))
opt.input_label_h5 = os.path.join(opt.input_dir, 'caption_{x}_label.h5'.format(x= relative_str))
opt.checkpoint_path = os.path.join(opt.output_dir, 'save_' + relative_str)

opt.beam_size = 2
opt.seq_per_img = 1
opt.batch_size = 32
opt.max_epochs = 50
opt.save_checkpoint_every = 1000

if not os.path.isdir(opt.output_dir): os.mkdir(opt.output_dir)
if not os.path.isdir(opt.checkpoint_path): os.mkdir(opt.checkpoint_path)

print('parsed input parameters:')
params_to_print = vars(opt)  # convert to ordinary dict
print(json.dumps(params_to_print, indent=2))
train(opt)
