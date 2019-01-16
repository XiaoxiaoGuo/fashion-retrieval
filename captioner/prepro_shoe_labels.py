"""
Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import re

def sent_to_tok(s):
    sent = ''.join([i for i in s if not i.isdigit() and not i == '.' and not i == '\\' and not i == '/'])
    sent = sent.replace(',', ' , ')
    sent = re.sub('\s\s+', ' ', sent)

    tok = sent.split(' ')
    tok = [t for t in tok if len(t) > 0]

    return tok

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for img in imgs:
    sents= img['caption']
    tokens = []
    for sent in sents:
        tok = sent_to_tok(sent)
        tokens.extend(tok)

    for w in tokens:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  f = open('vocab', 'w')
  for count, w in cw:
      f.write(w + ' ' + str(count) + '\n')
  f.close()

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
      sents = img['caption']
      for sent in sents:
          txt = sent.split(' ')
          nw = len(txt)
          sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count >0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')

  for img in imgs:
    img['final_captions'] = []
    sents = img['caption']
    for sent in sents:
        txt = sent.split(' ')
        caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_captions'].append(caption)


  return vocab

def encode_captions(imgs, params, wtoi):

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1

    counter += n

  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length

def main(params):


    output_json = os.path.join(params['output_dir'], 'caption_relative.json')
    output_h5 = os.path.join(params['output_dir'], 'caption_relative_label.h5')
    data = json.load(open(params['input_json'], 'r'))

    for iData in range(len(data)):
        data[iData]['caption'] = data[iData]['relative_caption']

    seed(123) # make reproducible

    # create the vocab
    vocab = build_vocab(data, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(data, params, wtoi)

    # create output h5 file
    f_lb = h5py.File(output_h5, "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file

    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(data):

        jimg = {}
        jimg['split'] = img['split']

        # jimg['file_path'] = os.path.join(params['image_root'], img['image_id'][0]) # copy it over, might need

        im_A_id = img['image_id'][0][4:-4]
        im_B_id = img['image_id'][1][4:-4]
        jimg['id'] = im_A_id + '_' + im_B_id

        out['images'].append(jimg)


    json.dump(out, open(output_json, 'w'))
    print('wrote ', output_json)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input_json = 'relativeCaptionAMTdata/relative_caption_batch123_denoised.json'
    # image_root = '/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/imgs'

    # input_json = 'relativeCaptionAMTdata/augmentation.json'
    # image_root = '/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/imgs_aug'

    input_json = 'relativeCaptionAMTdata/augx4.json'
    image_root = '/dccstor/foodvr1/fashionProjects/fashionDialog/data/amt_3k/imgs_augx4'

    parser.add_argument('--output_dir', default='debug_output_augx4', type=str, help='temp output folder')
    parser.add_argument('--input_json', default=input_json, type=str, help='AMT annotated data, including relative and direct captions')
    parser.add_argument('--image_root', default=image_root, type=str)

    # additional caption label parameters
    parser.add_argument('--max_length', default=16, type=int,help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=4*4, type=int,help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()

    print('parsed input parameters:')
    params = vars(args)  # convert to ordinary dict
    print(json.dumps(params, indent=2))

    main(params)




