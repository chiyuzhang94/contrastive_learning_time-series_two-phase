# LSTM

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.embed import TimeFeatureEmbedding
from torch.nn import init

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, e_layer, pred_leng, mask_rate, freq, contrastive_loss=0.0, dropout=0.1):
        super(LSTM, self).__init__()
        print("Mask Rate:", mask_rate)
        print("CL loss lambda", contrastive_loss)
        
        self.contrastive_loss = contrastive_loss
        self.enc_embedding = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=e_layer, batch_first=True, dropout=dropout)

        self.decoder = nn.Linear(hidden_size, output_size, bias=True)
        self.drop = nn.Dropout(dropout)
        self.pred_leng = pred_leng
        self.mask_rate = mask_rate
        self.init_weights()

    def init_weights(self):
        self.enc_embedding.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)
        for layer_p in self.encoder._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.normal(self.encoder.__getattr__(p), 0.0, 0.01)
                elif 'bias':
                    init.constant(self.encoder.__getattr__(p), 0.0)

    def forward(self, x):
        # print("x", x.shape)
        enc_embed = self.drop(self.enc_embedding(x))

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out, _ = self.encoder(enc_embed)
        y = self.decoder(enc_out)

        # enc_out_aug = None
        if self.contrastive_loss > 0.0:
            # masking augmentation
            mask = generate_binomial_mask(enc_embed.size(0), enc_embed.size(1), p=self.mask_rate).to(enc_embed.device)
            enc_embed_aug = enc_embed.clone()
            enc_embed_aug[~mask] = 0
            enc_out_aug, _ = self.encoder(enc_embed_aug)
        else:
            enc_out_aug = None


        return y[:,-self.pred_leng:,:], enc_out, enc_out_aug