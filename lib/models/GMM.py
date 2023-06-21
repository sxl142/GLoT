import torch
import os.path as osp
import torch.nn as nn
import os
from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor
from lib.models.transformer_global import Transformer
import torch.nn.functional as F
import importlib
class GMM(nn.Module):
    def __init__(
            self,
            seqlen,
            n_layers=1,
            d_model=2048,
            num_head=8, 
            dropout=0., 
            drop_path_r=0.,
            atten_drop=0.,
            mask_ratio=0.,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(GMM, self).__init__()
            
        self.proj = nn.Linear(2048, d_model)
        self.trans = Transformer(depth=n_layers, embed_dim=d_model, \
                mlp_hidden_dim=d_model*4, h=num_head, drop_rate=dropout, \
                drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=seqlen)
        self.out_proj = nn.Linear(d_model // 2, 2048)
        self.mask_ratio = mask_ratio
        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()
        self.initialize_weights()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, is_train=False, J_regressor=None):
        batch_size, seqlen = input.shape[:2]

        input = self.proj(input)
        if is_train:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=True, mask_ratio=self.mask_ratio)
        else:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=False, mask_ratio=0.)
        pred = self.trans.forward_decoder(mem, ids_restore)  # [N, L, p*p*3]

        if is_train:
            feature = self.out_proj(pred)
        else:
            feature = self.out_proj(pred)[:, seqlen // 2][:, None, :]

        smpl_output_global, pred_global = self.regressor(feature, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        scores = None
        if is_train:
            size = seqlen
        else:
            size = 1

        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(batch_size, size, -1)
            s['verts'] = s['verts'].reshape(batch_size, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, size, -1, 3, 3)
            s['scores'] = scores
       
        return smpl_output_global, mask_ids, mem, pred_global

    def initialize_weights(self):
        torch.nn.init.normal_(self.trans.pos_embed, std=.02)
        torch.nn.init.normal_(self.trans.decoder_pos_embed, std=.02)
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
