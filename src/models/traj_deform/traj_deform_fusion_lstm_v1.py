import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

# def KL(mu, logvar):
#     mu = mu.view(mu.shape[0], -1)
#     logvar = logvar.view(logvar.shape[0], -1)
#     loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
#     # high star implementation
#     # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
#     loss = torch.mean(loss)
#     return loss

def KL(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    kl_loss = torch.mean(kl_loss)
    return kl_loss


class PointNet2ClsSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[3, 32, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 512], use_xyz=True
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 4), # class num = 4
        )
    
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        c_in = 4
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=32,
                mlp=[c_in, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + c_in, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
    
    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])
    
# CVAE decoder
class LSTMDecoder(nn.Module):
    def __init__(self, num_layers=2, input_dim=41, hidden_dim=128, num_steps=40, wpt_dim=9):
        super(LSTMDecoder, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.pos_decoder = nn.Linear(self.hidden_size, self.wpt_dim)

    def forward(self, input_feat):

        batch_size = input_feat.shape[0]

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_feat.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_feat.device)

        lstm_out, (h, c) = self.lstm(input_feat, (h, c))
        out_traj_offset = self.pos_decoder(lstm_out)

        return out_traj_offset

class TrajDeformFusionLSTM(nn.Module):
    def __init__(self, pcd_feat_dim=32, cp_feat_dim=32,  
                        hidden_dim=128, 
                        num_steps=30, wpt_dim=9, decoder_layers=1,
                        lbd_cls=0.1, lbd_affordance=0.1, lbd_dir=1.0, lbd_deform=1.0, train_traj_start=10000, dataset_type=0):
        super(TrajDeformFusionLSTM, self).__init__()

        self.rot_dim = 6 if wpt_dim == 9 else 3
        self.pcd_feat_dim = pcd_feat_dim
        self.cp_feat_dim = cp_feat_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim
        self.decoder_layers = decoder_layers

        self.pointnet2cls = PointNet2ClsSSG({'feat_dim': pcd_feat_dim})
        self.pointnet2seg = PointNet2SemSegSSG({'feat_dim': pcd_feat_dim})

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        self.affordance_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(pcd_feat_dim, 1, kernel_size=1)
        )

        self.sigmoid = torch.nn.Sigmoid()

        #############################################
        # =========== for rotation head =========== #
        #############################################

        self.rotation_head = nn.Sequential(
            nn.Linear(pcd_feat_dim + cp_feat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.rot_dim)
        )

        ###########################################################
        # =========== for trajectory deformation head =========== #
        ###########################################################

        # encode_point_feat_dim = 6
        # self.all_encoder = nn.Sequential(
        #     nn.Conv1d(pcd_feat_dim, encode_point_feat_dim, kernel_size=1)
        # )

        self.mlp_cp = nn.Linear(3, cp_feat_dim)
        self.lstm_decoder = LSTMDecoder(
                                num_layers=decoder_layers, 
                                input_dim=pcd_feat_dim*2+wpt_dim, 
                                hidden_dim=hidden_dim, 
                                num_steps=num_steps, wpt_dim=wpt_dim
                            ) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')
        self.CELoss = nn.CrossEntropyLoss(reduction='mean')

        self.train_traj_start = train_traj_start
        
        self.lbd_cls = lbd_cls
        self.lbd_affordance = lbd_affordance
        self.lbd_dir = lbd_dir
        self.lbd_deform = lbd_deform

        self.dataset_type = dataset_type # 0 for absolute, 1 for residule

    # input sz bszx3x2
    def rot6d_to_rotmat(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps) # Rds[i, i] = the product of Rgts[i] and Rps[i]
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta) # theta = 1 will be 0 (the best)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.rot6d_to_rotmat(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.rot6d_to_rotmat(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta
    
    #########################################
    # ===================================== #
    # =========== loss function =========== #
    # ===================================== #
    #########################################

    def get_loss(self, iter, pcs, affordance, difficulty, temp_traj, traj):
        batch_size = traj.shape[0]

        difficulty_pred, affordance_pred, rotation_pred, traj_deform_offset_pred = self.forward(iter, pcs, temp_traj, traj) 

        deform_loss = torch.Tensor([0]).to(pcs.device)
        dir_loss = torch.Tensor([0]).to(pcs.device)

        ###################################################
        # =========== for classification head =========== #
        ###################################################

        difficulty_oh = F.one_hot(difficulty, num_classes=4).double().to(pcs.device)
        cls_loss = self.CELoss(difficulty_pred, difficulty_oh)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance_loss = F.binary_cross_entropy_with_logits(affordance_pred, affordance.unsqueeze(1))

        if iter < self.train_traj_start:
            losses = {}
            losses['cls'] = cls_loss
            losses['afford'] = affordance_loss
            losses['deform'] = deform_loss
            losses['dir'] = dir_loss

            losses['total'] = affordance_loss
            return losses

        #############################################
        # =========== for rotation head =========== #
        #############################################

        input_dir = traj[:, 0, 3:] # 9d
        dir_loss = self.get_6d_rot_loss(rotation_pred, input_dir).mean()

        ###########################################################
        # =========== for trajectory deformation head =========== #
        ###########################################################

        # trajectory deformation loss

        # for position offset
        first_pos_diff = traj[:, 0, :3] - temp_traj[:, 0, :3] # find the first-wpt difference
        if self.dataset_type == 0: # absolute
            temp_traj[:, :, :3] += first_pos_diff.unsqueeze(1) # align template traj to target traj via first wpt
        elif self.dataset_type == 1: # residual
            temp_traj[:, 0, :3] += first_pos_diff # align template traj to target traj via first wpt
        pos_offset = traj[:, :, :3] - temp_traj[:, :, :3]

        # for rotation offset
        temp_rotmat_xy = temp_traj[:, :, 3:].reshape(batch_size, self.num_steps, 2, 3)
        temp_rotmat_z = torch.cross(temp_rotmat_xy[:, :, 0], temp_rotmat_xy[:, :, 1]).unsqueeze(2)
        temp_rotmat = torch.cat([temp_rotmat_xy, temp_rotmat_z], dim=2) # row vector (the inverse of the temp rotation matrix)
        temp_rotmat = temp_rotmat.reshape(-1, 3, 3)

        rotmat_xy = traj[:, :, 3:].reshape(batch_size, self.num_steps, 2, 3) # column vector 
        rotmat_z = torch.cross(rotmat_xy[:, :, 0], rotmat_xy[:, :, 1]).unsqueeze(2)
        rotmat = torch.cat([rotmat_xy, rotmat_z], dim=2).permute(0, 1, 3, 2) # col vector (the inverse of the temp rotation matrix)
        rotmat = rotmat.reshape(-1, 3, 3)
        rotmat_offset = torch.bmm(rotmat, temp_rotmat).permute(0, 2, 1).reshape(-1, 9)[:, :6]

        # compute loss
        deform_wps_pos = traj_deform_offset_pred[:, :, :3]
        deform_pos_loss = self.MSELoss(deform_wps_pos.reshape(batch_size, self.num_steps * 3), 
                                       pos_offset.reshape(batch_size, self.num_steps * 3))
        
        deform_wps_rot = traj_deform_offset_pred[:, :, 3:]
        deform_wps_rot = deform_wps_rot.reshape(-1, 6)
        deform_rot_loss = self.get_6d_rot_loss(deform_wps_rot, 
                                               rotmat_offset)
        
        deform_rot_loss = deform_rot_loss.mean()
        deform_loss = deform_pos_loss + deform_rot_loss

        losses = {}
        losses['cls'] = cls_loss
        losses['afford'] = affordance_loss
        losses['dir'] = dir_loss
        losses['deform'] = deform_loss

        losses['total'] = self.lbd_cls * cls_loss + \
                            self.lbd_affordance * affordance_loss + \
                            self.lbd_dir * dir_loss + \
                            self.lbd_deform * deform_loss

        return losses


    ############################################
    # ======================================== #
    # =========== forward function =========== #
    # ======================================== #
    ############################################

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, pcs_feat: B x F x N
    def forward(self, iter, pcs, temp_traj, traj):

        batch_size = pcs.shape[0]

        contact_point = traj[:, 0, :3]

        temp_traj_clone = temp_traj.clone()
        pcd_input_length = pcs.shape[1] + temp_traj.shape[1]
        pn_input = torch.zeros((batch_size, pcd_input_length, 7)).to(pcs.device)

        temp_traj_align_offset = temp_traj[:, 0, :3] - contact_point
        if self.dataset_type == 0: # absolute
            temp_traj_clone[:, :, :3] -= temp_traj_align_offset.unsqueeze(1)
        elif self.dataset_type == 1: # residual
            temp_traj_clone[:, 0, :3] -= temp_traj_align_offset
        pn_input[:, :pcs.shape[1], :6] = pcs.repeat(1, 1, 2)
        pn_input[:, pcs.shape[1]:, :6] = temp_traj_clone[:, :, :3].repeat(1, 1, 2)
        pn_input[:, pcs.shape[1]:, 6] = 1
        
        whole_feats = self.pointnet2seg(pn_input)

        ###################################################
        # =========== for classification head =========== #
        ###################################################

        pcs_repeat = pcs.repeat(1, 1, 2)
        pointnet2cls_out = self.pointnet2cls(pcs_repeat)
        difficulty = F.log_softmax(pointnet2cls_out, -1)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance = self.affordance_head(whole_feats[:, :, :pcs.shape[1]])
        if iter < self.train_traj_start:
            return difficulty, affordance, None, None 
        
        #######################################################################
        # =========== extract shape feature using affordance head =========== #
        #######################################################################
        
        # TODO: may need to encode whole_feats to lower dimensional features

        pn_feat_dim = whole_feats.shape[1]
        affordance_sigmoid = self.sigmoid(affordance)
        affordance_min = torch.unsqueeze(torch.min(affordance_sigmoid, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance_sigmoid, dim=2).values, 1)
        affordance_norm = (affordance_sigmoid - affordance_min) / (affordance_max - affordance_min)
        part_cond = torch.where(affordance_norm > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = torch.zeros((batch_size, pn_feat_dim)).to(pcs.device)
        max_iter = torch.max(part_cond0) + 1
        for i in range(max_iter):

            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            tmp_max = torch.max(whole_feats[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max

        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)
        f_traj = whole_feats[:, :, pcs.shape[1]:].permute(0, 2, 1) # (batch, traj_length, feat_dim)
        
        #############################################
        # =========== for rotation head =========== #
        #############################################
        
        rot_input = torch.cat([f_s, f_cp], dim=-1)
        rotation = self.rotation_head(rot_input)

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        f_s_repeat = f_s.unsqueeze(1).repeat(1, self.num_steps, 1)
        f_all = torch.cat([f_s_repeat, f_traj, temp_traj_clone], dim=-1)
        traj_deform_offset = self.lstm_decoder(f_all)

        return difficulty, affordance, rotation, traj_deform_offset
    
    def sample(self, pcs, template_info, difficulty=None, use_gt_cp=False, return_feat=False):

        batch_size = pcs.shape[0]

        ###################################################
        # =========== for classification head =========== #
        ###################################################

        pcs_repeat = pcs.repeat(1, 1, 2)
        if difficulty is not None:
            target_difficulty = difficulty
        else :
            pointnet2cls_out = self.pointnet2cls(pcs_repeat)
            difficulty = F.log_softmax(pointnet2cls_out, -1)
            target_difficulty = torch.argmax(difficulty, dim=-1)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        pn_input = torch.zeros((batch_size, pcs.shape[1], 7)).to(pcs.device)
        pn_input[:, :, :6] = pcs_repeat
        pcs_feats = self.pointnet2seg(pn_input)
        affordance = self.affordance_head(pcs_feats)
        affordance = self.sigmoid(affordance) # Todo: remove comment

        if use_gt_cp == True:
            contact_point = pcs[:, 0]
        else :
            affordance_min = torch.unsqueeze(torch.min(affordance, dim=2).values, 1)
            affordance_max = torch.unsqueeze(torch.max(affordance, dim=2).values, 1)
            affordance = (affordance - affordance_min) / (affordance_max - affordance_min)
            contact_cond = torch.where(affordance == torch.max(affordance)) # only high response region selected
            contact_cond0 = contact_cond[0].to(torch.long) # point cloud id
            contact_cond2 = contact_cond[2].to(torch.long) # contact point ind for the point cloud
            contact_point = pcs[contact_cond0, contact_cond2]

        ########################################################
        # =========== get the target template info =========== #
        ########################################################

        temp_traj = None
        for i, current_cls in enumerate(target_difficulty):
            c = current_cls.int().item()
            if temp_traj is None:
                temp_traj = template_info[c][0].unsqueeze(0)
            else :
                traj = template_info[c][0].unsqueeze(0)
                temp_traj = torch.cat([temp_traj, traj], dim=0)

        temp_traj_clone = temp_traj.clone()
        pcd_input_length = pcs.shape[1] + temp_traj_clone.shape[1]
        pn_input = torch.zeros((batch_size, pcd_input_length, 7)).to(pcs.device)

        temp_traj_align_offset = temp_traj[:, 0, :3] - contact_point
        if self.dataset_type == 0: # absolute
            temp_traj_clone[:, :, :3] -= temp_traj_align_offset.unsqueeze(1)
        elif self.dataset_type == 1: # residual
            temp_traj_clone[:, 0, :3] -= temp_traj_align_offset
        pn_input[:, :pcs.shape[1], :6] = pcs.repeat(1, 1, 2)
        pn_input[:, pcs.shape[1]:, :6] = temp_traj[:, :, :3].repeat(1, 1, 2)
        pn_input[:, pcs.shape[1]:, 6] = 1

        #######################################################################
        # =========== extract shape feature using affordance head =========== #
        #######################################################################

        whole_feats = self.pointnet2seg(pn_input)
        part_score = self.affordance_head(whole_feats[:, :, :pcs.shape[1]])
        part_score = self.sigmoid(part_score) # Todo: remove comment

        # choose 10 features from segmented point cloud
        part_score_min = torch.unsqueeze(torch.min(part_score, dim=2).values, 1)
        part_score_max = torch.unsqueeze(torch.max(part_score, dim=2).values, 1)
        part_score = (part_score - part_score_min) / (part_score_max - part_score_min)

        part_cond = torch.where(part_score > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = whole_feats[:, :, 0].clone()
        for i in range(batch_size):
            
            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            tmp_max = torch.max(whole_feats[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max

        # f_s = torch.randn(whole_feats_part.shape).to(whole_feats_part.device)
        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)
        f_traj = whole_feats[:, :, pcs.shape[1]:].permute(0, 2, 1) # (batch, traj_length, feat_dim)

        #############################################
        # =========== for rotation head =========== #
        #############################################

        rot_input = torch.cat([f_s, f_cp], dim=-1)
        contact_rotation = self.rotation_head(rot_input)

        ##############################################################
        # =========== for trajectory deformstruction head =========== #
        ##############################################################

        f_s_repeat = f_s.unsqueeze(1).repeat(1, self.num_steps, 1)
        f_all = torch.cat([f_s_repeat, f_traj, temp_traj_clone], dim=-1)
        traj_deform_offset = self.lstm_decoder(f_all)
        
        # for position offset
        ret_traj = torch.zeros((batch_size, self.num_steps, self.wpt_dim))
        ret_traj[:, :, :3] = temp_traj_clone[:, :, :3] + traj_deform_offset[:, :, :3]

        # for rotation offset
        temp_rotmat = self.rot6d_to_rotmat(temp_traj_clone[:, :, 3:].reshape(-1, 2, 3).permute(0, 2, 1))
        offset_rotmat = self.rot6d_to_rotmat(traj_deform_offset[:, :, 3:].reshape(-1, 2, 3).permute(0, 2, 1))

        bmm_res = torch.bmm(offset_rotmat, temp_rotmat).permute(0, 2, 1).reshape(batch_size, self.num_steps, 3, 3)
        ret_traj[:, :, 3:] = bmm_res.reshape(batch_size, self.num_steps, 9)[:, :, :6] # only the first two column vectors in the rotation matrix used

        if return_feat:
            return target_difficulty, contact_point, part_score, ret_traj, whole_feats_part
        return target_difficulty, contact_point, part_score, ret_traj