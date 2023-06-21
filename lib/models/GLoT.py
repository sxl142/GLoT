import torch
import os.path as osp
import torch.nn as nn
import os
from lib.core.config import BASE_DATA_DIR
from lib.models.HSCR import HSCR
from lib.models.GMM import GMM 
from lib.models.transformer_local import Transformer as local_transformer_encoder
import torch.nn.functional as F
from lib.models.trans_operator import CrossAttention, Mlp

class GLoT(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            d_model=2048,
            num_head=8, 
            dropout=0., 
            drop_path_r=0.,
            atten_drop=0.,
            mask_ratio=0.,
            short_n_layers = 3,
            short_d_model = 512,
            short_num_head = 8,
            short_dropout = 0.1, 
            short_drop_path_r = 0.2,
            short_atten_drop = 0.,
            stride_short=4,
            drop_reg_short=0.5,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(GLoT, self).__init__()
        self.stride_short = stride_short
        self.mid_frame = seqlen // 2
        self.short_n_layers = short_n_layers

        self.proj_short = nn.Linear(2048, short_d_model)
        self.proj_mem = nn.Linear(d_model, short_d_model)
        self.local_trans_de = CrossAttention(short_d_model, num_heads=8, qkv_bias=False, \
        qk_scale=None, attn_drop=0., proj_drop=0.)
        self.local_trans_en = local_transformer_encoder(depth=short_n_layers, embed_dim=short_d_model, \
                mlp_hidden_dim=short_d_model*4, h=short_num_head, drop_rate=short_dropout, \
                drop_path_rate=short_drop_path_r, attn_drop_rate=short_atten_drop, length=self.stride_short * 2 + 1)
        self.regressor = HSCR(drop=drop_reg_short)
        self.initialize_weights()
       
        self.global_modeling = GMM(seqlen, n_layers, d_model, num_head, dropout, drop_path_r, atten_drop, mask_ratio)
        

    def forward(self, input, is_train=False, J_regressor=None):
        batch_size = input.shape[0]
        smpl_output_global, mask_ids, mem, pred_global = self.global_modeling(input, is_train=is_train, J_regressor=J_regressor)
        
        x_short = input[:, self.mid_frame - self.stride_short:self.mid_frame + self.stride_short + 1]
        x_short = self.proj_short(x_short)
        x_short = self.local_trans_en(x_short)
        mid_fea = x_short[:, self.stride_short - 1: self.stride_short + 2]
        mem = self.proj_mem(mem)
        out_short = self.local_trans_de(mid_fea, mem)
        # print(out_short.shape)
        if is_train:
            feature = out_short
        else:
            feature = out_short[:, 1][:, None, :]

        smpl_output = self.regressor(feature, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
        scores = None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, size, -1)
                s['verts'] = s['verts'].reshape(batch_size, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, mask_ids, smpl_output_global



    def initialize_weights(self):
        torch.nn.init.normal_(self.local_trans_en.pos_embed, std=.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

