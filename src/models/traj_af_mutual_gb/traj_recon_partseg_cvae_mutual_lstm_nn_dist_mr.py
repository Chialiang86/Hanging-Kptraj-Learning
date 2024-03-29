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
    def __init__(self, pcd_feat_dim=128, traj_feat_dim=256, cp_feat_dim=32, hidden_dim=128, z_feat_dim=64):
        super(AllEncoder, self).__init__()

        self.mlp1 = nn.Linear(pcd_feat_dim + cp_feat_dim + traj_feat_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, z_feat_dim)
        self.get_mu = nn.Linear(z_feat_dim, z_feat_dim)
        self.get_logvar = nn.Linear(z_feat_dim, z_feat_dim)

        self.z_dim = z_feat_dim

    # pcs_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, traj_feat, cp_feat):
        net = torch.cat([pn_feat, traj_feat, cp_feat], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar

# CVAE decoder
class AllDecoder(nn.Module):
    def __init__(self, pcd_feat_dim, cp_feat_dim=32, z_feat_dim=64, hidden_dim=128, num_steps=30, wpt_dim=6):
        super(AllDecoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(pcd_feat_dim + cp_feat_dim + z_feat_dim, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, num_steps * wpt_dim)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(pcd_feat_dim + cp_feat_dim + z_feat_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_steps * wpt_dim)
        )
        
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

    # pn_feat B x F, query_fats: B x 6
    # output: B
    def forward(self, pn_feat, cp_feat, z_all):
        batch_size = z_all.shape[0]
        x = torch.cat([pn_feat, cp_feat, z_all], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.num_steps, 6)
        return x

class TrajReconPartSegMutual(nn.Module):
    def __init__(self, pcd_feat_dim=256, traj_feat_dim=128, cp_feat_dim=32,  
                        hidden_dim=128, z_feat_dim=64, 
                        num_steps=30, wpt_dim=6,
                        lbd_kl=1.0, lbd_recon=1.0, lbd_dir=1.0, kl_annealing=0, train_traj_start=10000, dataset_type=0):
        super(TrajReconPartSegMutual, self).__init__()

        self.z_dim = z_feat_dim

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pcd_feat_dim})

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        self.affordance_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(pcd_feat_dim, 1, kernel_size=1)
        )

        self.sigmoid = torch.nn.Sigmoid()

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        self.mlp_traj = TrajEncoder(traj_feat_dim=traj_feat_dim, num_steps=num_steps, wpt_dim=wpt_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim) # contact point
        self.all_encoder = AllEncoder(
                                pcd_feat_dim=pcd_feat_dim, traj_feat_dim=traj_feat_dim, cp_feat_dim=cp_feat_dim,
                                hidden_dim=hidden_dim, z_feat_dim=z_feat_dim
                            ) # CVAE encoder
        self.all_decoder = AllDecoder(
                                pcd_feat_dim=pcd_feat_dim, cp_feat_dim=cp_feat_dim,
                                z_feat_dim=z_feat_dim, hidden_dim=hidden_dim, 
                                num_steps=num_steps, wpt_dim=wpt_dim
                            ) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')

        self.train_traj_start = train_traj_start

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
    # pred_result_logits: B, pcs_feat: B x F x N
    def forward(self, iter, pcs, traj, contact_point):

        pcs_repeat = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_repeat)

        affordance = self.affordance_head(whole_feats)
        affordance_sigmoid = self.sigmoid(affordance)
        if iter < self.train_traj_start:
            return affordance, None, None, None, None
        
        # choose 10 features from segmented point cloud
        affordance_min = torch.unsqueeze(torch.min(affordance_sigmoid, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance_sigmoid, dim=2).values, 1)
        affordance_norm = (affordance_sigmoid - affordance_min) / (affordance_max - affordance_min)
        part_cond = torch.where(affordance_norm > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = whole_feats[:, :, 0].clone()
        max_iter = torch.max(part_cond0) + 1
        pcs_part_list = []
        for i in range(max_iter):
            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            tmp_max = torch.max(whole_feats[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            # pcs_part = pcs[i, part_cond2[cond]] # choose the sub point cloud that affordance score > threshold
            # ind = furthest_point_sample(pcs_part.unsqueeze(0).contiguous(), self.num_steps).long().reshape(-1) # get the 10 indexes from the sub point cloud using furthest point sampling
            # point_ind = part_cond2[cond][ind]
            # tmp_max = torch.max(whole_feats[i, :, point_ind], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max
            pcs_part_list.append(pcs[i, part_cond2[cond]])

        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)
        f_traj = self.mlp_traj(traj)

        z_all, mu, logvar = self.all_encoder(f_s, f_traj, f_cp)
        recon_traj = self.all_decoder(f_s, f_cp, z_all)

        return affordance, pcs_part_list, recon_traj, mu, logvar

    def sample(self, pcs):
        batch_size = pcs.shape[0]
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        pcs_input = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs_input)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance = self.affordance_head(whole_feats)
        # affordance = self.sigmoid(affordance) # Todo: remove comment

        affordance_min = torch.unsqueeze(torch.min(affordance, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance, dim=2).values, 1)
        affordance = (affordance - affordance_min) / (affordance_max - affordance_min)
        contact_cond = torch.where(affordance == torch.max(affordance)) # only high response region selected
        contact_cond0 = contact_cond[0].to(torch.long) # point cloud id
        contact_cond2 = contact_cond[2].to(torch.long) # contact point ind for the point cloud

        contact_point = pcs[contact_cond0, contact_cond2]

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        # choose 10 features from segmented point cloud
        part_cond = torch.where(affordance > 0.3) # only high response region selected
        part_cond0 = part_cond[0].to(torch.long)
        part_cond2 = part_cond[2].to(torch.long)
        whole_feats_part = whole_feats[:, :, 0].clone()
        max_iter = torch.max(part_cond0) + 1
        for i in range(max_iter):
            cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
            pcs_part = pcs[i, part_cond2[cond]] # choose the sub point cloud that affordance score > threshold
            ind = furthest_point_sample(pcs_part.unsqueeze(0).contiguous(), self.num_steps).long().reshape(-1) # get the 10 indexes from the sub point cloud using furthest point sampling
            point_ind = part_cond2[cond][ind]
            tmp_max = torch.max(whole_feats[i, :, point_ind], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
            whole_feats_part[i] = tmp_max

        f_s = whole_feats_part
        f_cp = self.mlp_cp(contact_point)
        recon_traj = self.all_decoder(f_s, f_cp, z_all)
        ret_traj = torch.zeros(recon_traj.shape)
        if self.dataset_type == 0: # absolute 
            ret_traj = recon_traj
            ret_traj[:, 0, :3] = contact_point

        if self.dataset_type == 1: # residual 
            ret_traj[:, 0, :3] = contact_point

            recon_dir = recon_traj[:, 0]
            recon_dirmat = self.rot6d_to_rotmat(recon_dir.reshape(-1, 2, 3).permute(0, 2, 1))
            recon_rotvec = R.from_matrix(recon_dirmat.cpu().detach().numpy()).as_rotvec()
            ret_traj[:, 0, 3:] = torch.from_numpy(recon_rotvec)

            ret_traj[:, 1:] = recon_traj[:, 1:]

        return affordance, ret_traj

    def get_nn_loss(self, pcs_part_list : list, traj : torch.Tensor):
        # similar to nearest 1 neighbor
        # reference: https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695

        traj_abso = traj[:, :, :3].clone()
        if self.dataset_type == 1: # residual 
            # recover position from residual to absolute
            for wpt_id in range(1, traj.shape[1]):
                traj_abso[:, wpt_id] += traj_abso[:, wpt_id - 1]
        
        batch_size = traj_abso.shape[0]
        traj_unsqueeze = traj_abso[:, :self.num_steps, :3].unsqueeze(2) # (B x T x 3) => (B x T x 1 x 3)

        mean_min_dists = torch.zeros((self.num_steps, 1)).to('cuda')
        for i in range(batch_size):
            pcs_unsqueeze = pcs_part_list[i].unsqueeze(0).repeat(self.num_steps, 1, 1) # (N x 3) => (T x N x 3)
            diff = pcs_unsqueeze - traj_unsqueeze[i] # (T x N x 3)
            dist = torch.norm(diff, dim=2) # (T x N)
            mean_min_dist = dist.topk(1, largest=False).values # smallest value  # (T x 1)
            mean_min_dists = mean_min_dists + mean_min_dist

        return torch.mean(mean_min_dists)
    
    def get_loss(self, iter, pcs, traj, contact_point, affordance, lbd_kl=1.0):
        batch_size = traj.shape[0]

        affordance_pred, pcs_part_list, recon_traj, mu, logvar = self.forward(iter, pcs, traj, contact_point)

        nn_loss = torch.Tensor([0]).to('cuda')
        recon_loss = torch.Tensor([0]).to('cuda')
        dir_loss = torch.Tensor([0]).to('cuda')
        kl_loss = torch.Tensor([0]).to('cuda')

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        affordance_loss = F.binary_cross_entropy_with_logits(affordance_pred, affordance.unsqueeze(1))
        if iter < self.train_traj_start:
            losses = {}
            losses['afford'] = affordance_loss
            losses['nn'] = nn_loss
            losses['kl'] = kl_loss
            losses['recon'] = recon_loss
            losses['dir'] = dir_loss
            losses['total'] = affordance_loss
            return losses

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        recon_loss = torch.Tensor([0]).to('cuda')
        dir_loss = torch.Tensor([0]).to('cuda')
        dist_loss = torch.Tensor([0]).to('cuda')
        nn_loss = torch.Tensor([0]).to('cuda')

        if self.dataset_type == 0: # absolute 
            recon_wps = recon_traj
            input_wps = traj
            
            nn_loss = self.get_nn_loss(pcs_part_list, recon_wps)

            mean_interval_gt = torch.mean(torch.norm(traj[:, 1:, :3] - traj[:, :-1, :3], dim=2))
            mean_interval_recon = torch.mean(torch.norm(recon_wps[:, 1:, :3] - recon_wps[:, :-1, :3], dim=2))
            dist_loss = torch.abs(mean_interval_gt - mean_interval_recon) / mean_interval_gt
        
            bound1 = int(0.25 * self.num_steps)
            bound2 = int(0.5 * self.num_steps)
            recon_loss_0 = self.MSELoss(recon_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim), input_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim))
            recon_loss_1 = self.MSELoss(recon_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim), input_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim))
            recon_loss_2 = self.MSELoss(recon_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) *  self.wpt_dim), input_wps[:, bound2:].view(batch_size, (self.num_steps - bound2) * self.wpt_dim))
            recon_loss = recon_loss_0 + 0.5 * recon_loss_1 + 0.25 * recon_loss_2

        if self.dataset_type == 1: # residual
            input_dir = traj[:, 0, :]
            recon_dir = recon_traj[:, 0, :]
            input_wps = traj[:, 1:, :]
            recon_wps = recon_traj[:, 1:, :]

            # recon loss
            bound1 = int(0.25 * self.num_steps)
            bound2 = int(0.5 * self.num_steps)
            end = self.num_steps - 1
            recon_loss_0 = self.MSELoss(recon_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim), input_wps[:, :bound1].view(batch_size, bound1 * self.wpt_dim))
            recon_loss_1 = self.MSELoss(recon_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim), input_wps[:, bound1:bound2].view(batch_size, (bound2 - bound1) * self.wpt_dim))
            recon_loss_2 = self.MSELoss(recon_wps[:, bound2:].view(batch_size, (end - bound2) *  self.wpt_dim), input_wps[:, bound2:].view(batch_size, (end - bound2) * self.wpt_dim))
            wpt_loss = recon_loss_0 + 0.5 * recon_loss_1 + 0.25 * recon_loss_2

            dir_loss = self.get_6d_rot_loss(recon_dir, input_dir)
            dir_loss = dir_loss.mean()

            recon_loss = self.lbd_dir * dir_loss + wpt_loss

            # dist loss
            mean_interval_gt = torch.mean(torch.norm(traj[:, 1:, :3], dim=2))
            mean_interval_recon = torch.mean(torch.norm(recon_wps[:, 1:, :3], dim=2))
            dist_loss = torch.abs(mean_interval_gt - mean_interval_recon) / mean_interval_gt
            
            # nn loss
            nn_loss = self.get_nn_loss(pcs, recon_traj)

        kl_loss = KL(mu, logvar)
        losses = {}
        losses['afford'] = affordance_loss
        losses['nn'] = nn_loss
        losses['kl'] = kl_loss
        losses['recon'] = recon_loss
        losses['dir'] = dir_loss
        losses['dist'] = dist_loss

        if self.kl_annealing == 0:
            losses['total'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon + 0.001 * nn_loss + 0.1 * dist_loss + 0.1 * affordance_loss
        elif self.kl_annealing == 1:
            losses['total'] = kl_loss * lbd_kl + recon_loss * self.lbd_recon + 0.001 * nn_loss + 0.1 * dist_loss + 0.1 * affordance_loss

        return losses