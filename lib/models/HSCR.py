import numpy as np
import torch
import torch.nn as nn
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.models.spin import projection
from lib.utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

ANCESTOR_INDEX = [
    [],
    [0], 
    [0], 
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 1, 4],
    [0, 2, 5],
    [0, 3, 6],
    [0, 1, 4, 7],
    [0, 2, 5, 8],
    [0, 3, 6, 9], 
    [0, 3, 6, 9], 
    [0, 3, 6, 9],
    [0, 3, 6, 9, 12],
    [0, 3, 6, 9, 13],
    [0, 3, 6, 9, 14],
    [0, 3, 6, 9, 13, 16],
    [0, 3, 6, 9, 14, 17],
    [0, 3, 6, 9, 13, 16, 18],
    [0, 3, 6, 9, 14, 17, 19],
    [0, 3, 6, 9, 13, 16, 18, 20],
    [0, 3, 6, 9, 14, 17, 19, 21]
]

class KTD(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(KTD, self).__init__()
        npose_per_joint = 6
        nshape = 10
        ncam = 3
        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(ANCESTOR_INDEX):
            regressor = nn.Linear(hidden_dim + npose_per_joint * len(ancestor_idx) + 6, npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)
        self.decshape = nn.Linear(hidden_dim, nshape)
        self.deccam = nn.Linear(hidden_dim, ncam)

    def forward(self, x, global_pose):
        pose = []
        cnt = 0
        for ancestor_idx, reg in zip(ANCESTOR_INDEX, self.joint_regs):
            
            ances = torch.cat([x] + [pose[i] for i in ancestor_idx], dim=-1)
            ances = torch.cat((ances, global_pose[:, :, cnt: cnt + 6]), dim=-1)

            cnt += 1
            pose.append(reg(ances))
            
        pred_pose = torch.cat(pose, dim=-1)
        return pred_pose

class HSCR(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, hidden_dim=1024, drop=0.5):
        super(HSCR, self).__init__()
        npose = 24 * 6
        self.fc1 = nn.Linear(256 + 10, hidden_dim)
        self.fc2 = nn.Linear(256 + npose, hidden_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim * 2 + 3, 3)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        self.local_reg = KTD(hidden_dim)

    def forward(self, x, init_pose, init_shape, init_cam, is_train=False, J_regressor=None):
        pred_pose = init_pose.detach()
        pred_shape = init_shape.detach()
        pred_cam = init_cam.detach()
        
        xc_shape_cam = torch.cat([x, pred_shape], -1)
        xc_pose_cam = torch.cat([x, pred_pose], -1)

        xc_shape_cam = self.fc1(xc_shape_cam)
        xc_shape_cam = self.drop1(xc_shape_cam)

        xc_pose_cam = self.fc2(xc_pose_cam)
        xc_pose_cam = self.drop2(xc_pose_cam)
       
        pred_pose = self.local_reg(xc_pose_cam, pred_pose) + pred_pose
        pred_shape = self.decshape(xc_shape_cam) + pred_shape  
        pred_cam = self.deccam(torch.cat([xc_pose_cam, xc_shape_cam, pred_cam], -1)) + pred_cam

        pred_pose = pred_pose.reshape(-1, 144)
        pred_shape = pred_shape.reshape(-1, 10)
        pred_cam = pred_cam.reshape(-1, 3)
        batch_size = pred_pose.shape[0]

        out_put = self.get_output(pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor)
        return out_put

    def get_output(self, pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor):
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'rotmat' : pred_rotmat
        }]
        return output