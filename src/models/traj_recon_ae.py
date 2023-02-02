import torch
import torch.nn as nn
import torch.nn.functional as F

# def KL(mu, logvar):
#     mu = mu.view(mu.shape[0], -1)
#     logvar = logvar.view(logvar.shape[0], -1)
#     # loss = 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - 1 - logvar, 1)
#     # high star implementation
#     # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
#     loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), 1)
#     loss = torch.mean(loss)
#     return loss

def KL(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    kl_loss = torch.mean(kl_loss)
    return kl_loss

# AE encoder
class ALLEncoder(nn.Module):
    def __init__(self, hidden_dim=64, z_feat_dim=32, num_steps=30, wpt_dim=6):
        super(ALLEncoder, self).__init__()
        
        self.num_steps = num_steps
        self.z_dim = z_feat_dim
        self.wpt_dim = wpt_dim

        self.mlp1 = nn.Linear(num_steps * wpt_dim, hidden_dim)
        # self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, z_feat_dim)

    # pcs_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, traj_feats):
        batch_size = traj_feats.shape[0]
        z = F.leaky_relu(self.mlp1(traj_feats.view(batch_size, self.num_steps * self.wpt_dim)))
        # z = F.leaky_relu(self.mlp2(z))
        z = self.mlp3(z)

        return z

# VAE decoder
class AllDecoder(nn.Module):
    def __init__(self, z_feat_dim=32, hidden_dim=64, num_steps=30, wpt_dim=6):
        super(AllDecoder, self).__init__()

        self.z_feat_dim = z_feat_dim
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim
        self.mlp1 = nn.Linear(z_feat_dim, hidden_dim)
        # self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, num_steps * wpt_dim)

    # pcs_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, z_all):
        batch_size = z_all.shape[0]
        x = self.mlp1(z_all)
        # x = self.mlp2(x)
        x = self.mlp3(x)
        x = x.view(batch_size, self.num_steps, self.wpt_dim)
        return x

class TrajRecon(nn.Module):
    def __init__(self, hidden_dim=64, z_feat_dim=256, num_steps=30, wpt_dim=6, lbd_recon=1.0, dataset_type=0):
        super(TrajRecon, self).__init__()

        # Not in use
        # self.dir_encoder = nn.Linear(6, dir_feat_dim) # TODO: may be used in the future
        
        self.all_encoder = ALLEncoder(hidden_dim=hidden_dim, z_feat_dim=z_feat_dim, num_steps=num_steps) # CVAE encoder
        self.all_decoder = AllDecoder(z_feat_dim=z_feat_dim, hidden_dim=hidden_dim, num_steps=num_steps, wpt_dim=wpt_dim) # CVAE decoder
        self.MSELoss = nn.MSELoss(reduction='mean')

        self.z_dim = z_feat_dim
        self.num_steps = num_steps
        self.wpt_dim = wpt_dim

        self.lbd_recon = lbd_recon
        
        self.dataset_type = dataset_type # 0 for absolute, 1 for residule

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, pcs_feats: B x F x N
    def forward(self, traj):

        z_all = self.all_encoder(traj)
        recon_traj = self.all_decoder(z_all)

        return recon_traj

    def sample(self, batch_size):
        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        recon_traj = self.all_decoder(z_all)

        return recon_traj

    def sample_n(self, batch_size, rvs=100):
        z_all = torch.Tensor(torch.randn(batch_size * rvs, self.z_dim)).cuda()

        recon_traj = self.all_decoder(z_all)

        return recon_traj
    
    def get_loss(self, traj):
        batch_size = traj.shape[0]
        recon_traj = self.forward(traj)
        
        if self.dataset_type == 0: # absolute
            recon_wps = recon_traj
            input_wps = traj
            recon_loss = self.MSELoss(recon_wps.view(batch_size, self.num_steps * 6), input_wps.view(batch_size, self.num_steps * 6))

        if self.dataset_type == 1: # residual
            recon_absolute = recon_traj[:, 0, :]
            recon_residual = recon_traj[:, 1:, :]
            input_absolute = traj[:, 0, :]
            input_residual = traj[:, 1:, :]
            recon_absolute_loss = self.MSELoss(recon_absolute.view(batch_size, 1 * 6), input_absolute.view(batch_size, 1 * 6))
            recon_residual_loss = self.MSELoss(recon_residual.view(batch_size, (self.num_steps - 1) * 6), input_residual.view(batch_size, (self.num_steps - 1) * 6))
            recon_loss = recon_absolute_loss * 100 + recon_residual_loss

        losses = {}
        losses['total'] = recon_loss

        return losses

    # def get_loss_test_rotation(self, traj):
    #     batch_size = traj.shape[0]
    #     recon_traj, mu, logvar = self.forward(traj)
    #     # recon_dir = recon_traj[:, 0, :]
    #     # recon_wps = recon_traj[:, 1:, :]
    #     # input_dir = traj[:, 0, :]
    #     # input_wps = traj[:, 1:, :]
    #     # recon_xyz_loss = self.MSELoss(recon_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 0:3].contiguous().view(batch_size, (self.num_steps - 1) * 3))
    #     # recon_rotation_loss = self.MSELoss(recon_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3), input_wps[:, :, 3:6].contiguous().view(batch_size, (self.num_steps - 1) * 3))
    #     # recon_loss = recon_xyz_loss.mean() + recon_rotation_loss.mean() * 100

    #     recon_wps = recon_traj
    #     input_wps = traj
    #     recon_xyz_loss = self.MSELoss(recon_wps[:, :, 0:3].contiguous().view(batch_size, self.num_steps * 3), input_wps[:, :, 0:3].contiguous().view(batch_size, self.num_steps * 3))
    #     recon_rotation_loss = self.MSELoss(recon_wps[:, :, 3:6].contiguous().view(batch_size, self.num_steps * 3), input_wps[:, :, 3:6].contiguous().view(batch_size, self.num_steps * 3))
    #     recon_loss = recon_xyz_loss + recon_rotation_loss * 100

    #     kl_loss = KL(mu, logvar)
    #     losses = {}
    #     losses['kl'] = kl_loss
    #     losses['recon'] = recon_loss
    #     losses['recon_xyz'] = recon_xyz_loss
    #     losses['recon_rotation'] = recon_rotation_loss
    #     losses['total'] = kl_loss * self.lbd_kl + recon_loss * self.lbd_recon

    #     return losses
    
