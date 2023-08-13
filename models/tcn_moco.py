#TCN
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.embed import TimeFeatureEmbedding
import sys, math, random, copy
from typing import Union, Callable, Optional, List
from models.weight_norm import WeightNorm
from models.data_aug import Data_Aug, generate_binomial_mask
from models.tcn_encoder import *
from models.losses import hierarchical_contrastive_loss
from einops import reduce, rearrange, repeat
from models.embed import DataEmbedding 
from models.embed import TimeFeatureEmbedding
from models.dtcn_encoder import *
from models.lstm_encoder import LSTM_Encoder
from models.informer_encoder import Informer_Encoder

class TCNBase(nn.Module):
    def __init__(self, model_name, input_size, represent_size, e_layer, freq, 
                 hidden_size=64, kernel_size = 3, dropout = 0.1, mask_rate=0.5, mare=False):
        super(TCNBase, self).__init__()
        print("[INFO] Mask Rate:", mask_rate)
        # print("CL loss lambda", contrastive_loss)
        
        self.mare = mare
        # self.contrastive_loss = contrastive_loss
        self.enc_embedding = nn.Linear(input_size, hidden_size)

        if "dtcn" in model_name:
            print("[INFO] USE DTCN")
            self.encoder = DilatedConvEncoder(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size)
        elif "lstm" in model_name:
            print("[INFO] USE LSTM")
            self.encoder = LSTM_Encoder(hidden_size, represent_size, e_layer, hidden_size)
        elif "informer" in model_name:
            print("[INFO] USE Informer")
            self.enc_embedding = nn.Linear(input_size, 128)
            self.encoder = Informer_Encoder(represent_size, d_model=128, e_layer = e_layer, d_ff=hidden_size, dropout=dropout)
        else:
            print("[INFO] USE TCN")
            self.encoder = TemporalConvNet(hidden_size, [hidden_size] * e_layer + [represent_size], kernel_size, dropout=dropout)

        if self.mare:
            print("[INFO] Use MARE.")
            self.kernels = [1, 2, 4, 8, 16, 32, 64, 128]
            self.tfd = nn.ModuleList(
                [nn.Conv1d(represent_size, represent_size, k, padding=k-1) for k in self.kernels]
            )
        else:
            self.tfd = nn.Linear(represent_size, represent_size)

        self.drop = nn.Dropout(dropout)
        # self.pred_leng = pred_leng
        self.mask_rate = mask_rate
        # self.init_weights()

    def init_weights(self):
        self.enc_embedding.weight.data.normal_(0, 0.01)

    def forward(self, x, mask_flag=False):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        enc_embed = self.enc_embedding(x)

        if mask_flag:
            # masking augmentation
            mask = generate_binomial_mask(enc_embed.size(0), enc_embed.size(1), p=self.mask_rate).to(enc_embed.device)
            # mask &= nan_mask
            
        else:
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)

        mask &= nan_mask
        enc_embed[~mask] = 0

        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        enc_out = self.encoder(enc_embed.transpose(1, 2)).transpose(1, 2)


        if self.mare:
            enc_out = enc_out.transpose(1, 2)
            # print("enc_out before", enc_out.shape)
            trend = []
            for idx, mod in enumerate(self.tfd):
                out = mod(enc_out)  # b d t
                if self.kernels[idx] != 1:
                    out = out[..., :-(self.kernels[idx] - 1)]
                trend.append(out.transpose(1, 2))  # b t d
            
            enc_out = reduce(
                rearrange(trend, 'list b t d -> list b t d'),
                'list b t d -> b t d', 'mean'
            )
        else:
            enc_out = self.tfd(enc_out)

        return enc_out

class TCN_MoCo_Pretrain(nn.Module):
    def __init__(self, model_name,
                 input_size, hidden_size, output_size, e_layer, pred_leng, freq,
                 kernel_size = 3, dropout = 0.1, l2norm = False, average_pool = False, data_aug = None, tempral_cl = False,
                 device: Optional[str] = 'cuda', mask_rate = 0.5, moco_cl_weight = 1.0, mare = False,
                 K: Optional[int] = 256,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 1.0):
        super(TCN_MoCo_Pretrain, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.l2norm = l2norm
        self.average_pool = average_pool
        self.data_aug = data_aug
        self.tempral_cl = tempral_cl
        self.moco_cl_weight = moco_cl_weight

        # if self.data_aug == "cost":
        #     print("[INFO] Use Data augmentation method:", self.data_aug)
        #     self.data_aug_tool = Data_Aug(sigma=0.5, p=0.5)

        print("[INFO] l2norm is ", str(self.l2norm))

        self.encoder_q = TCNBase(model_name, input_size, hidden_size, e_layer, freq, kernel_size=kernel_size, dropout = dropout, mask_rate = mask_rate, mare =mare).to(self.device)
        self.encoder_k = copy.deepcopy(self.encoder_q)


        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.head_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(hidden_size, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    #     self.decoder = nn.Linear(hidden_size, output_size, bias=True)
    #     self.drop = nn.Dropout(dropout)
    #     self.pred_leng = pred_leng
    #     self.init_weights()


    # def init_weights(self):
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.normal_(0, 0.01)

    def compute_loss(self, q, k, k_negs):
        
        if self.l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            k_negs = F.normalize(k_negs, dim=-1)
            
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k=None, pretrain=True):

        if pretrain:
            rand_idx = np.random.randint(0, x_q.shape[1])
            # q_x = self.data_aug_tool.cost_transform(x)
            enc_out = self.encoder_q(x_q, mask_flag=True)
            
            if enc_out is not None:
                if self.average_pool:
                    q_t = self.head_q(torch.mean(enc_out, 1))
                else:
                    q_t = self.head_q(enc_out[:,rand_idx])

            # compute key features
            with torch.no_grad():  # no gradient for keys
                self._momentum_update_key_encoder()  # update key encoder

                # if self.data_aug:
                #     x_k= self.data_aug_tool.cost_transform(x)

                k_t = self.encoder_k(x_k, mask_flag=True)
                
                if k_t is not None:
                    if self.average_pool: 
                        k_t = self.head_k(torch.mean(k_t, 1))
                    else:
                        k_t = self.head_k(k_t[:,rand_idx])

            cl_loss = 0.0

            cl_loss += self.compute_loss(q_t, k_t, self.queue.clone().detach()) * self.moco_cl_weight
            self._dequeue_and_enqueue(k_t)

            if self.tempral_cl: ## moco+hierachical cl
                aug_x = self.encoder_k(x)
                cl_loss += hierarchical_contrastive_loss(q_t, k_t, self.l2norm) * (1.0 - self.moco_cl_weight)
            
            return None, enc_out, cl_loss
        else:
            enc_out = self.encoder_q(x_q, mask_flag=False)
            # enc_out = self.head_q(enc_out)
            
            return None, enc_out, None

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class TCN_MoCo_Task(nn.Module):
    def __init__(self, 
                 hidden_size, output_size, pred_leng, encoder, dropout = 0.1, freeze_encoder= False,
                 device: Optional[str] = 'cuda'):
        super(TCN_MoCo_Task, self).__init__()
        
        
        self.device = device
        
        self.encoder = encoder

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.drop = nn.Dropout(dropout)
        self.pred_leng = pred_leng

        self.decoder.apply(weights_init)
        
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "head_q" not in name:
                    param.requires_grad = False


    # def init_weights(self):
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.normal_(0, 0.02)

    def forward(self, x):
        _, enc_out, _ = self.encoder(x, pretrain=False)

        y = self.decoder(enc_out)

        return y[:,-self.pred_leng:,:], enc_out, None
