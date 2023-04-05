import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
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


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
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
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
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

# CVAE encoder
class AllEncoder(nn.Module):
    def __init__(self, pcd_feat_dim=128, traj_feat_dim=128, cp_feat_dim=64, hidden_dim=128, z_feat_dim=64):
        super(AllEncoder, self).__init__()

        self.mlp1 = nn.Linear(pcd_feat_dim + traj_feat_dim + cp_feat_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, z_feat_dim)
        self.get_mu = nn.Linear(z_feat_dim, z_feat_dim)
        self.get_logvar = nn.Linear(z_feat_dim, z_feat_dim)

        self.z_dim = z_feat_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, wpt_feats, cp_feat):
        batch_size = pn_feat.shape[0]

        traj_feat = wpt_feats.reshape(batch_size, -1) # flatten
        x = torch.cat([pn_feat, traj_feat, cp_feat], dim=-1)
        x = F.leaky_relu(self.mlp1(x))
        x = self.mlp2(x)
        mu = self.get_mu(x)
        logvar = self.get_logvar(x)
        noise = torch.Tensor(torch.randn(*mu.shape)).to(pn_feat.device)
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar
    
# CVAE decoder
class LSTMDecoder(nn.Module):
    def __init__(self, num_layers=1, pcd_feat_dim=128, wpt_feat_dim=32, z_feat_dim=64, hidden_dim=128, num_steps=40, wpt_dim=9):
        super(LSTMDecoder, self).__init__()
        self.input_size = wpt_feat_dim + pcd_feat_dim + z_feat_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.wpt_dim = wpt_dim
        self.num_steps = num_steps

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.pos_decoder = nn.Linear(self.hidden_size, self.wpt_dim)

    def forward(self, pn_feat, wpt_feats, z_all):

        batch_size = wpt_feats.shape[0]

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(wpt_feats.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(wpt_feats.device)

        out = torch.zeros((batch_size, self.num_steps - 1, self.hidden_size)).to(wpt_feats.device)

        for i in range(0, self.num_steps - 1):
            wpt_feat = wpt_feats[:, i]
            x = torch.cat([wpt_feat, pn_feat, z_all], dim=-1).unsqueeze(1).to(wpt_feats.device)
            out_tmp, (h, c) = self.lstm(x, (h, c))
            out[:, i] = out_tmp.squeeze()

        out = self.pos_decoder(out)
        return out
    
    def inference(self, first_wpt, wpt_encoder, pn_feat, z_all):
        
        batch_size = first_wpt.shape[0]

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(first_wpt.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(first_wpt.device)

        out = torch.zeros((batch_size, self.num_steps, self.wpt_dim)).to(first_wpt.device)
        out[:, i] = first_wpt

        for i in range(1, self.num_steps):
            wpt_feat = wpt_encoder(out[:, i - 1])
            x = torch.cat([wpt_feat, pn_feat, z_all], dim=-1).unsqueeze(1).to(first_wpt.device)
            out_h, (h, c) = self.lstm(x, (h, c))
            out[:, i] = self.pos_decoder(out_h)

        return out

class TrajReconPartSegMutualLSTM(nn.Module):
    def __init__(self, pcd_feat_dim=128, wpt_feat_dim=32, cp_feat_dim=32,  
                        hidden_dim=128, z_feat_dim=64, 
                        num_steps=30, wpt_dim=9, decoder_layers=1,
                        lbd_kl=1.0, lbd_recon=1.0, lbd_dir=1.0, kl_annealing=0, train_traj_start=10000, dataset_type=0):
        super(TrajReconPartSegMutualLSTM, self).__init__()

        self.rot_dim = 6 if wpt_dim == 9 else 3
        self.z_dim = z_feat_dim
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pcd_feat_dim})

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

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        self.mlp_cp = nn.Linear(3, cp_feat_dim)
        self.mlp_wpt = nn.Linear(wpt_dim, wpt_feat_dim)
        self.all_encoder = AllEncoder(
                                pcd_feat_dim=pcd_feat_dim, traj_feat_dim=(wpt_feat_dim * num_steps), cp_feat_dim=cp_feat_dim,
                                hidden_dim=hidden_dim, z_feat_dim=z_feat_dim
                            ) # CVAE encoder
        self.lstm_decoder = LSTMDecoder(
                                num_layers=decoder_layers, 
                                pcd_feat_dim=pcd_feat_dim, wpt_feat_dim=wpt_feat_dim, z_feat_dim=z_feat_dim, 
                                hidden_dim=hidden_dim, 
                                num_steps=num_steps, wpt_dim=wpt_dim
                            ) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')

        self.train_traj_start = train_traj_start

        self.lbd_kl = lbd_kl
        self.lbd_recon = lbd_recon
        self.lbd_dir = lbd_dir
        self.kl_annealing = kl_annealing

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

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, pcs_feat: B x F x N
    def forward(self, iter, pcs, traj, contact_point):

        pcs_repeat = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_repeat)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance = self.affordance_head(whole_feats)
        if iter < self.train_traj_start:
            return affordance, None, None, None, None
        
        #######################################################################
        # =========== extract shape feature using affordance head =========== #
        #######################################################################
        
        affordance_sigmoid = self.sigmoid(affordance)
        # choose 10 features from segmented point cloud
        affordance_min = torch.unsqueeze(torch.min(affordance_sigmoid, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance_sigmoid, dim=2).values, 1)
        affordance_norm = (affordance_sigmoid - affordance_min) / (affordance_max - affordance_min)
        part_cond = torch.where(affordance_norm > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = whole_feats[:, :, 0].clone()
        max_iter = torch.max(part_cond0) + 1
        for i in range(max_iter):

            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            tmp_max = torch.max(whole_feats[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max

        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)
        f_wpts = self.mlp_wpt(traj)

        #############################################
        # =========== for rotation head =========== #
        #############################################

        rot_input = torch.cat([f_s, f_cp], dim=-1)
        rotation = self.rotation_head(rot_input)

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        # def forward(self, first_wpt, pcd_feat, z):
        z_all, mu, logvar = self.all_encoder(f_s, f_wpts, f_cp)
        recon_traj = self.lstm_decoder(f_s, f_wpts, z_all)

        return affordance, rotation, recon_traj, mu, logvar

    def get_loss(self, iter, pcs, traj, contact_point, affordance, lbd_kl=1.0):
        batch_size = traj.shape[0]

        affordance_pred, rotation, recon_traj, mu, logvar = self.forward(iter, pcs, traj, contact_point) # recon_traj contain all trajectories from 2 ~ T waypoints

        recon_loss = torch.Tensor([0]).to(pcs.device)
        dir_loss = torch.Tensor([0]).to(pcs.device)
        kl_loss = torch.Tensor([0]).to(pcs.device)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance_loss = F.binary_cross_entropy_with_logits(affordance_pred, affordance.unsqueeze(1))
        if iter < self.train_traj_start:
            losses = {}
            losses['afford'] = affordance_loss
            losses['kl'] = kl_loss
            losses['recon'] = recon_loss
            losses['dir'] = dir_loss

            losses['total'] = affordance_loss
            return losses

        recon_loss = torch.Tensor([0]).to(pcs.device) 
        dir_loss = torch.Tensor([0]).to(pcs.device)
        kl_loss = torch.Tensor([0]).to(pcs.device)

        #############################################
        # =========== for rotation head =========== #
        #############################################

        input_dir = traj[:, 0, 3:] # 9d
        dir_loss = self.get_6d_rot_loss(rotation, input_dir)
        dir_loss = dir_loss.mean()

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        # trajectory reconstruction loss
        loss_wpt_num = self.num_steps - 1
        input_wps_pos = traj[:, 1:, :3]
        recon_wps_pos = recon_traj[:, :, :3]
        recon_pos_loss = self.MSELoss(recon_wps_pos.reshape(batch_size, loss_wpt_num * 3), input_wps_pos.reshape(batch_size, loss_wpt_num * 3))
        input_wps_rot = traj[:, 1:, 3:] 
        recon_wps_rot = recon_traj[:, :, 3:]
        recon_rot_loss = self.get_6d_rot_loss(recon_wps_rot, input_wps_rot)
        recon_rot_loss = recon_rot_loss.mean()
        recon_loss = recon_pos_loss + recon_rot_loss

        # kl regularization loss
        kl_loss = KL(mu, logvar)

        losses = {}
        losses['afford'] = affordance_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['kl'] = kl_loss

        if self.kl_annealing == 0:
            losses['total'] = self.lbd_kl * kl_loss + self.lbd_recon * recon_loss + self.lbd_dir * dir_loss + 0.1 * affordance_loss
        elif self.kl_annealing == 1:
            losses['total'] = lbd_kl * kl_loss + self.lbd_recon * recon_loss + self.lbd_dir * dir_loss + 0.1 * affordance_loss

        return losses
    
    def sample(self, pcs):
        batch_size = pcs.shape[0]

        pcs_input = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_input)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance = self.affordance_head(whole_feats)
        affordance = self.sigmoid(affordance)

        affordance_min = torch.unsqueeze(torch.min(affordance, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance, dim=2).values, 1)
        affordance = (affordance - affordance_min) / (affordance_max - affordance_min)
        contact_cond = torch.where(affordance == torch.max(affordance)) # only high response region selected
        contact_cond0 = contact_cond[0].to(torch.long) # point cloud id
        contact_cond2 = contact_cond[2].to(torch.long) # contact point ind for the point cloud

        contact_point = pcs[contact_cond0, contact_cond2]

        #######################################################################
        # =========== extract shape feature using affordance head =========== #
        #######################################################################

        # choose 10 features from segmented point cloud
        part_cond = torch.where(affordance > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = whole_feats[:, :, 0].clone()
        max_iter = torch.max(part_cond0) + 1
        for i in range(max_iter):
            
            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            tmp_max = torch.max(whole_feats[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max

        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)

        #############################################
        # =========== for rotation head =========== #
        #############################################

        rot_input = torch.cat([f_s, f_cp], dim=-1)
        contact_rotation = self.rotation_head(rot_input)

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).to(pcs.device)

        first_wpt = torch.cat([contact_point, contact_rotation], dim=-1)
        recon_traj = self.lstm_decoder.inference(first_wpt, self.mlp_wpt, f_s, z_all)
        
        ret_traj = torch.zeros((batch_size, self.num_steps, 6))
        recon_dirmat = self.rot6d_to_rotmat(recon_traj[:, :, 3:].reshape(-1, 2, 3).permute(0, 2, 1))
        recon_rotvec = R.from_matrix(recon_dirmat.cpu().detach().numpy()).as_rotvec()
        recon_rotvec = torch.from_numpy(recon_rotvec).to(pcs.device)
        ret_traj[:, :, 3:] = recon_rotvec.reshape(batch_size, self.num_steps, 3)

        return affordance, ret_traj