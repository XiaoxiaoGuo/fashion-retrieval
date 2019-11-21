from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


#
class NetSynUser(nn.Module):
    def __init__(self, num_emb):
        super(NetSynUser, self).__init__()
        hid_dim = 256
        self.hid_dim = hid_dim
        self.rep_dim = hid_dim

        self.emb_txt = torch.nn.Embedding(num_embeddings=num_emb, embedding_dim=hid_dim * 2 )
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim * 2)
        self.cnn_txt = torch.nn.Conv1d(in_channels=1, out_channels=hid_dim * 2, kernel_size=(2, hid_dim*2), bias=True)
        self.fc_txt = nn.Linear(in_features=hid_dim * 2, out_features=hid_dim, bias=False)
        self.img_linear = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=True)

        self.fc_joint = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        self.rnn = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.head = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)

    # fine-tuning the history tracker and policy part
    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False
        return

    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True
        return

    def forward_image(self, image_input):
        return self.img_linear(image_input)

    def forward_text(self, text_input):
        x = self.emb_txt(text_input).unsqueeze(1)
        x = self.cnn_txt(x)
        x, _ = torch.max(x, dim=2)
        x = x.squeeze()
        x = self.fc_txt(self.bn2(x))
        return x

    def forward(self, img_input, txt_input):
        x1 = self.forward_image(img_input)
        x2 = self.forward_text(txt_input)
        x = x1 + x2
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x

    def merge_forward(self, img_emb, txt_input):
        x2 = self.forward_text(txt_input)
        x = img_emb + x2
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x

    def init_hid(self, batch_size):
        self.hx = Variable(torch.Tensor(batch_size, self.hid_dim).zero_())
        return

    def detach_hid(self):
        self.hx = Variable(self.hx.data)
        return