import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

def KL(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    kl_loss = torch.mean(kl_loss)
    return kl_loss

class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
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


class TrajEncoder(nn.Module):
    def __init__(self, traj_feat_dim, num_steps=30, wpt_dim=6):

        # traj_feat_dim = 128 for VAT-Mart

        super(TrajEncoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(num_steps * wpt_dim, 128),
        #     nn.Linear(128, 128),
        #     nn.Linear(128, traj_feat_dim)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(num_steps * wpt_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, traj_feat_dim)
        )

        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.view(batch_size, self.num_steps * 6).dtype, type(x.view(batch_size, self.num_steps * 6)))
        x = self.mlp(x.view(batch_size, self.num_steps * self.wpt_dim))
        return x

# CVAE encoder
class AllEncoder(nn.Module):
    def __init__(self, pcd_feat_dim=128, traj_feat_dim=256, hidden_dim=128, z_feat_dim=64):
        super(AllEncoder, self).__init__()

        self.mlp1 = nn.Linear(pcd_feat_dim + traj_feat_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, z_feat_dim)
        self.get_mu = nn.Linear(z_feat_dim, z_feat_dim)
        self.get_logvar = nn.Linear(z_feat_dim, z_feat_dim)

        self.z_dim = z_feat_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, traj_feat):
        net = torch.cat([pn_feat, traj_feat], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar

# CVAE decoder
class AllDecoder(nn.Module):
    def __init__(self, pcd_feat_dim, z_feat_dim=64, hidden_dim=128, num_steps=30, wpt_dim=6):
        super(AllDecoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(pcd_feat_dim + cp_feat_dim + z_feat_dim, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, num_steps * wpt_dim)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(pcd_feat_dim + z_feat_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_steps * wpt_dim)
        )
        
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

    # pn_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, z_all):
        batch_size = z_all.shape[0]
        x = torch.cat([pn_feat, z_all], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.num_steps, 6)
        return x

class TrajReconAffordanceNoFP(nn.Module):
    def __init__(self, pcd_feat_dim=256, traj_feat_dim=128,  
                        hidden_dim=128, z_feat_dim=64, 
                        num_steps=30, wpt_dim=6,
                        lbd_kl=1.0, lbd_recon=1.0, lbd_dir=1.0, kl_annealing=0, dataset_type=0):
        super(TrajReconAffordanceNoFP, self).__init__()

        self.z_dim = z_feat_dim

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pcd_feat_dim})
        self.mlp_traj = TrajEncoder(traj_feat_dim=traj_feat_dim, num_steps=num_steps, wpt_dim=wpt_dim)

        # Not in use
        # self.dir_encoder = nn.Linear(6, dir_feat_dim) # TODO: may be used in the future

        self.all_encoder = AllEncoder(
                                pcd_feat_dim=pcd_feat_dim, traj_feat_dim=traj_feat_dim,
                                hidden_dim=hidden_dim, z_feat_dim=z_feat_dim
                            ) # CVAE encoder
        self.all_decoder = AllDecoder(
                                pcd_feat_dim=pcd_feat_dim,
                                z_feat_dim=z_feat_dim, hidden_dim=hidden_dim, 
                                num_steps=num_steps, wpt_dim=wpt_dim
                            ) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')

        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

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
    #       Note: contact point of the pcs should be at the origin 
    # traj: B x T x 6 (float)
    #       Note: the final waypoint should be at the origin 
    def forward(self, pcs, traj, contact_point):
        # pcs[:, 0] = contact_point

        # align the the contact point to the origin (for both point cloud and trajectory)
        traj_copy = traj.clone()
        pcs_copy = pcs.clone()
        pcs_copy[:, :, :3] = pcs_copy[:, :, :3] - contact_point.unsqueeze(1)
        if self.dataset_type == 0: # absolute 
            traj_copy[:, :, :3] = traj_copy[:, :, :3] - contact_point.unsqueeze(1)

        pcs_copy = pcs_copy.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_copy)

        f_s = whole_feats[:, :, 0]
        f_traj = self.mlp_traj(traj_copy)

        z_all, mu, logvar = self.all_encoder(f_s, f_traj)
        recon_traj = self.all_decoder(f_s, z_all)

        # align to original coordinate frame (for both point cloud and trajectory)
        if self.dataset_type == 0: # absolute 
            recon_traj[:, :, :3] = recon_traj[:, :, :3] + contact_point.unsqueeze(1)

        return recon_traj, mu, logvar

    def sample(self, pcs, contact_point):
        batch_size = pcs.shape[0]
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        pcs_copy = pcs.clone()
        pcs_copy[:, :, :3] = pcs_copy[:, :, :3] - contact_point.unsqueeze(1)
        pcs_copy = pcs_copy.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_copy)
        f_s = whole_feats[:, :, 0]

        recon_traj = self.all_decoder(f_s, z_all)
        ret_traj = torch.zeros(recon_traj.shape)
        if self.dataset_type == 0: # absolute 
            ret_traj = recon_traj
            ret_traj[:, :, :3] = ret_traj[:, :, :3] + contact_point.unsqueeze(1)

        if self.dataset_type == 1: # residual 
            ret_traj[:, 0, :3] = contact_point

            recon_dir = recon_traj[:, 0]
            recon_dir = recon_dir.reshape(-1, 2, 3).permute(0, 2, 1)
            recon_dirmat = self.rot6d_to_rotmat(recon_dir)
            recon_rotvec = R.from_matrix(recon_dirmat.cpu().detach().numpy()).as_rotvec()
            ret_traj[:, 0, 3:] = torch.from_numpy(recon_rotvec)

            ret_traj[:, 1:] = recon_traj[:, 1:]

        return ret_traj

    # def sample_n(self, pcs, batch_size, rvs=100):
    #     z_all = torch.Tensor(torch.randn(batch_size * rvs, self.z_dim)).cuda()

    #     pcs = pcs.repeat(1, 1, 2)
    #     pcs_feat = self.pointnet2(pcs)

    #     net = pcs_feat[:, :, 0]
    #     net = net.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)

    #     recon_traj = self.all_decoder(z_all, net)

    #     return recon_traj

    def get_loss(self, pcs, traj, contact_point, lbd_kl=1.0):
        batch_size = traj.shape[0]

        recon_traj, mu, logvar = self.forward(pcs, traj, contact_point)

        recon_loss = torch.Tensor([0])
        dir_loss = torch.Tensor([0])
        if self.dataset_type == 0: # absolute 
            recon_wps = recon_traj
            input_wps = traj
            recon_loss = self.MSELoss(recon_wps.view(batch_size, self.num_steps * self.wpt_dim), input_wps.view(batch_size, self.num_steps * self.wpt_dim))

        if self.dataset_type == 1: # residualrecon_dir = recon_traj[:, 0, :]
            input_dir = traj[:, 0, :]
            recon_dir = recon_traj[:, 0, :]
            dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
            dir_loss = dir_loss.mean()

            input_wps = traj[:, 1:, :]
            recon_wps = recon_traj[:, 1:, :]
            wpt_loss = self.MSELoss(recon_wps.view(batch_size, (self.num_steps - 1) * self.wpt_dim), input_wps.view(batch_size, (self.num_steps - 1) * self.wpt_dim))
            
            recon_loss = self.lbd_dir * dir_loss + wpt_loss
            # recon_absolute = recon_traj[:, 0, :]
            # recon_residual = recon_traj[:, 1:, :]
            # input_absolute = traj[:, 0, :]
            # input_residual = traj[:, 1:, :]
            # recon_absolute_loss = self.MSELoss(recon_absolute.view(batch_size, 1 * 6), input_absolute.view(batch_size, 1 * 6))
            # recon_residual_loss = self.MSELoss(recon_residual.view(batch_size, (self.num_steps - 1) * 6), input_residual.view(batch_size, (self.num_steps - 1) * 6))
            # recon_loss = recon_absolute_loss * 100 + recon_residual_loss
        
        # recon_dir = recon_traj[:, 0, :]
        # input_dir = traj[:, 0, :]
        # dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
        # dir_loss = dir_loss.mean()

        kl_loss = KL(mu, logvar)
        losses = {}
        losses['dir'] = dir_loss
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss

        if self.kl_annealing == 0:
            losses['total'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon
        elif self.kl_annealing == 1:
            losses['total'] = kl_loss * lbd_kl + recon_loss * self.lbd_recon

        return losses

    # def get_loss_test_rotation(self, pcs, traj, batch_size):
    #     recon_traj, mu, logvar = self.forward(pcs, traj)
    #     recon_dir = recon_traj[:, 0, :]
    #     recon_wps = recon_traj[:, 1:, :]
    #     input_dir = traj[:, 0, :]
    #     input_wps = traj[:, 1:, :]
    #     recon_xyz_loss = self.MSELoss(recon_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3))
    #     recon_rotation_loss = self.MSELoss(recon_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3))
    #     recon_loss = recon_xyz_loss.mean() + recon_rotation_loss.mean() * 100

    #     dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
    #     dir_loss = dir_loss.mean()
    #     kl_loss = KL(mu, logvar)
    #     losses = {}
    #     losses['kl'] = kl_loss
    #     losses['recon'] = recon_loss
    #     losses['recon_xyz'] = recon_xyz_loss.mean()
    #     losses['recon_rotation'] = recon_rotation_loss.mean()
    #     losses['total'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon

    #     return losses

    # def inference_whole_pc(self, feat, dirs1, dirs2):
    #     num_pts = feat.shape[-1]
    #     batch_size = feat.shape[0]

    #     feat = feat.permute(0, 2, 1)  # B x N x F
    #     feat = feat.reshape(batch_size*num_pts, -1)

    #     input_queries = torch.cat([dirs1, dirs2], dim=-1)
    #     input_queries = input_queries.unsqueeze(dim=1).repeat(1, num_pts, 1)
    #     input_queries = input_queries.reshape(batch_size*num_pts, -1)

    #     pred_result_logits = self.critic(feat, input_queries)

    #     soft_pred_results = torch.sigmoid(pred_result_logits)
    #     soft_pred_results = soft_pred_results.reshape(batch_size, num_pts)

    #     return soft_pred_results

    # def inference(self, pcs, dirs1, dirs2):
    #     pcs = pcs.repeat(1, 1, 2)
    #     pcs_feat = self.pointnet2(pcs)

    #     net = pcs_feat[:, :, 0]

    #     input_queries = torch.cat([dirs1, dirs2], dim=1)

    #     pred_result_logits = self.critic(net, input_queries)

    #     pred_results = (pred_result_logits > 0)

    #     return pred_results
    
