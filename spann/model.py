# network structure
import math
from itertools import cycle

import numpy as np
import ot
import scanpy as sc
import scipy
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from easydl import AccuracyCounter, clear_output, one_hot, variable_to_numpy
from scipy.stats import f, norm, ttest_ind
from torch import nn as nn
from torch.autograd import Function
from torch.distributions import Beta, Normal, kl_divergence
from torch.nn import init
from torch.nn.parameter import Parameter


def kl_div(mu, var, weight=None):
    # calculate kl divergence
    loss = kl_divergence(Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))).sum(dim=1)

    # if weight is not None:
    #     loss = loss * weight.squeeze(dim=1)
    return loss.mean()


# def balanced_binary_cross_entropy(recon_x, x):

#     return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    # Returns the matrix of ||x_i-y_j||_p^p.

    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_gmm(mu_src: torch.Tensor, mu_dst: torch.Tensor, var_src: torch.Tensor, var_dst: torch.Tensor):
    # Calculate a Wasserstein distance matrix between the gmm distributions with diagonal variances

    std_src = var_src.sqrt()
    std_dst = var_dst.sqrt()
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)

    # distance_var = torch.sum(sum_matrix(std_src, std_dst) - 2 * (prod_matrix(std_src, std_dst) ** 0.5), 2)

    return distance_mean + distance_var


def calculate_bimodality_coefficient(series=None):
    """
    calculate BC-coefficient, higher BC coef means more likely to be a bimodal distribution
    """
    # Calculate skewness.
    # Correct for statistical sample bias.
    skewness = scipy.stats.skew(
        series,
        axis=None,
        bias=False,
    )
    # Calculate excess kurtosis.
    # Correct for statistical sample bias.
    kurtosis = scipy.stats.kurtosis(
        series,
        axis=None,
        fisher=True,
        bias=False,
    )
    # Calculate count.
    count = len(series)
    # Calculate count factor.
    count_factor = ((count - 1) ** 2) / ((count - 2) * (count - 3))
    # Count bimodality coefficient.
    coefficient = ((skewness**2) + 1) / (kurtosis + (3 * count_factor))
    # Return value.
    return coefficient


def dip(samples, num_bins=100, p=0.99, table=True):
    """
    dip test for distribution, to test is the distribution id multi-modality
    """
    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 5e-2

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[: xl + 1] - cdf[: xl + 1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:] - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl : xr + 1]
        idxs = idxs[xl : xr + 1]
        pdf = pdf[xl : xr + 1]

    if table:
        p_threshold, p_value = p_table(p, ans, samples.size, 10000)
        if ans < p_threshold:
            check = True
        return ans, p_threshold, check, p_value

    return ans


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1 - cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs(alpha[contact_points] - beta[contact_points])
    return diff.max(), diff


def p_table(p, ans, sample_size, n_samples):
    data = [np.random.randn(sample_size) for _ in range(n_samples)]
    dip_sample = [dip(samples, table=False) for samples in data]
    dips = np.hstack(dip_sample)
    dip_sample.append(ans)
    index = np.argsort(dip_sample)
    p_value = 1 - (np.argsort(index)[-1] + 1) / len(index)
    return np.percentile(dips, p * 100), p_value


### Networks
activation = {
    "relu": nn.ReLU(),
    "rrelu": nn.RReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "": None,
}


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization Layer
    :num_features: dimension of the features
    :n_domain: domain number

    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                # out[indices] = x[indices]
                self.bns[i].training = False
                out[indices] = self.bns[i](x[indices])
                self.bns[i].training = True
        return out


class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    :input_dim: dimension of input
    : output_dim: dimension of output
    :norm: batch normalization, '' represent no batch normalization, 1 represent regular batch normalization, int>1 represent domain-specific batch normalization of n domain
    :act: activation function, relu -> nn.ReLU, rrelu -> nn.RReLU, sigmoid -> nn.Sigmoid(), leaky_relu -> nn.LeakyReLU(), tanh -> nn.Tanh(),  '' -> None
    dropout: dropout rate
    """

    def __init__(self, input_dim, output_dim, norm="", act="", dropout=0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if type(norm) == int:
            if norm == 1:  # TO DO
                self.norm = nn.BatchNorm1d(output_dim)
            else:
                self.norm = DSBatchNorm(output_dim, norm)
        else:
            self.norm = None

        self.act = activation[act]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, y=None):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm.__class__.__name__ == "DSBatchNorm":
                h = self.norm(h, y)
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    :input_dim: input dimension
    :cfg: model structure configuration, example - [['fc', x_dim, n_domain, 'sigmoid']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i == 0:
                d_in = input_dim
            if layer[0] == "fc":
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)

    def forward(self, x, y=None):
        for layer in self.net:
            x = layer(x, y)
        return x


class Encoder(nn.Module):
    """
    VAE Encoder
    :input_dim: input dimension
    :cfg: encoder configuration, example - [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
    """

    def __init__(self, input_dim, cfg):
        super().__init__()

        enc = []
        mu_enc = []
        var_enc = []

        h_dim = cfg[-2][1]

        enc.append(NN(input_dim[0], cfg[:-1]))
        mu_enc.append(NN(h_dim, cfg[-1:]))
        var_enc.append(NN(h_dim, cfg[-1:]))

        self.enc = nn.ModuleList(enc)
        self.mu_enc = nn.ModuleList(mu_enc)
        self.var_enc = nn.ModuleList(var_enc)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, domain, y=None):
        """ """
        q = self.enc[domain](x, y)
        mu = self.mu_enc[domain](q, y)
        var = torch.exp(self.var_enc[domain](q, y))
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    """
    VAE Decoder
    :z_dim: latent dimension
    :cfg: decoder configuration, example - [['fc', adatas[i].obsm[obsm[i]].shape[1], 1, 'sigmoid']]

    """

    def __init__(self, z_dim, cfg):
        super().__init__()

        dec = []
        for i in cfg.keys():
            dec.append(NN(z_dim, cfg[i]))

        self.dec = nn.ModuleList(dec)

    def forward(self, z, domain, y=None):
        """ """
        reconx_x = self.dec[domain](z, y)

        return reconx_x


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """

    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """

    def __init__(self, in_dim, out_dim, hidden_mlp=1024, feat_dim=16, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, feat_dim)
        )
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer
    :num_classes (int): number of classes
    :epsilon (float): weight

    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1，再取log.

    def forward(self, inputs, targets):
        """
        :inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        :targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)  #

        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)  # 对行求和，消掉列，keepdim=True则列变成1
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


# optimal transport alignment
def ubot_CCD(sim, beta, device, stopThr=1e-4):
    """
    Unbalanced optimal transport function
    :sim: similarity matrix between two distributions
    :beta: marginal distribution of the transport object
    :return: pseudo label computed from the tranport plan, updated marginal distribution of the transport object, normarlized transport plan, confidence score of the transport plan

    """
    # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
    M = -sim
    alpha = ot.unif(sim.size(0))
    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(
        alpha, beta, M.detach().cpu().numpy(), reg=0.01, reg_m=0.5, stopThr=stopThr
    )
    Q_st = torch.from_numpy(Q_st).float().to(device)

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st / sum_pi
    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_st_bar, 1)

    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()

    return pseudo_label, new_beta, Q_st_bar, wt_i


def ensemble_score(logit_tensor, confidence_ot):
    """ """
    label_pred_tensor = nn.Softmax(-1)(logit_tensor)
    softmax_cls, _ = torch.max(label_pred_tensor, 1)
    entropy_cls = (label_pred_tensor * torch.log(label_pred_tensor)).sum(-1)
    entropy_cls = entropy_cls / np.log(label_pred_tensor.size()[1]) + 1

    ensemble = (entropy_cls + softmax_cls + confidence_ot) / 3
    return ensemble, entropy_cls, softmax_cls, confidence_ot


def mixup_target(x_batch, device):  ##batchsize*latent
    m = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
    x_batch_expand = x_batch.unsqueeze(1)  ##batchsize * 1 * input_dim
    weight = (m.sample([x_batch_expand.size()[0], x_batch_expand.size()[1]])).to(device)
    x_batch_mix = weight * x_batch + (1 - weight) * x_batch_expand  ##   batchsize * batchsize * input_dim
    x_batch_mix = x_batch_mix.view([-1, x_batch_mix.size()[-1]])  ## (batchsize * batchsize) * input_dim
    index = torch.arange(0, x_batch_mix.size()[0])  # (batchsize * batchsize)
    x_batch_mix = x_batch_mix[index % (x_batch.size()[0] + 1) != 0]  ##  (batchsize * (batchsize-1)) * input_dim
    return x_batch_mix


def pdists(A, squared=False, eps=0):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)

    if squared:
        return res
    else:
        res = res.clamp(min=eps).sqrt()
        return res


class SPANN_model:
    """
    SPANN model, using scRNA reference dataset to annotate spatial transcriptome dataset

    :x_dim: input dimension for the encoder
    :z_dim: latent feature dimension
    :enc: encoder parameters, for example, [['fc', 1024, 1, 'relu'], ['fc', 16, '', '']]
    :dec: decoder parameters, dec is a set {cm_dec_params, spa_dec_params, rna_dec_params}, each element shapes like enc
    :class_num: number of scRNA-seq cell types
    :device: device on which SPAMM model is aranged

    """

    def __init__(self, x_dim, z_dim, enc, dec, class_num, device):
        super().__init__()
        self.class_num = class_num
        self.device = device
        self.feat_dim = enc[-1][1]
        self.encoder = Encoder(x_dim, enc).to(device)
        self.decoder = Decoder(z_dim, dec).to(device)
        self.classifier = CLS(z_dim, class_num).to(device)

    def train(
        self,
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        spatial_coor,
        test_source_cm_dl,
        test_target_cm_dl,
        source_labels,
        cell_types,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
    ):
        """
        Training and evaluating function for SPANN applying on test spatial dataset with ground truth labels

        :source_cm_dl: torch dataloader of the common genes scRNA-seq data
        :target_cm_dl: torch dataloader of the common genes spatial data
        :source_sp_ds: torch dataset of the scRNA-seq specific genes data
        :target_sp_ds: torch dataset of the spatial specific genes data
        :spatial_coor: pandas dataframe of the raw spatial coordinates, column names ['X','Y']
        :test_source_cm_dl: torch dataloader of the common genes scRNA-seq data for test
        :test_target_cm_dl: torch dataloader of the common genes spatial data for test
        :source_labels: integer labels for scRNA-seq data
        :cell_types: list of all cell types exist in scRNA-seq dataset

        :lr: learning rate for VAE and classifier, default=1e-4
        :lamnda_recon: training weight for the reconstruction loss, default=2000
        :lambda_kl: training weight for the KL-divergence loss, default=0.5
        :lambda_spa: training weight for the adjacency loss, default=0.1, we recomend to set it smaller when the spatial expression pattern is weak
        :lambda_cd: training weight for the unbalanced optimal transport alignment loss, default=0.001, we recomend to set it higher when the gap between scRNA-seq and spatial datasets is big
        :lambda_nb: training weight for the neighbor loss, default=0.1
        :mu: updating speed of beta by moving average, default=0.5
        :resolution: the expected minimum proportion of known cells, 0 means no constraints on novel cell discovery and 1 means no novel cells, default=0.5
        :novel_cell_test: whether to apply dip test and compute BC co-efficient to adaptively select resolution, default=True
        :maxiter: maximum iteration, default=6000
        :miditer1: after which iteration SPANN starts to conduct UOT alignment, default=2000
        :miditer2: after which iteration SPANN starts to train with neighbor loss, default=5000
        :miditer3: after which iteration SPANN starts to train with adjacency loss, default=4000
        :test_freq: test frequency, default=1000

        :return: two AnnData objects, adata_source and adata_target, containing the spatial coordinates, latent embeddings, E-scores, predictions and etc.

        """
        optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=5e-4)

        target_coor = torch.tensor(np.array(spatial_coor)).to(self.device)
        target_coor = torch.tensor(target_coor, dtype=torch.float)
        target_coor = target_coor / torch.max(target_coor, axis=0)[0]
        beta = None
        beta_s = ot.unif(self.class_num)
        threshold = None
        iteration = 0

        while iteration < maxiter:
            if len(source_sp_ds) > len(target_sp_ds):
                iters = zip(source_cm_dl, cycle(target_cm_dl))
            else:
                iters = zip(cycle(source_cm_dl), target_cm_dl)
            for minibatch_id, ((im_source, label_source, id_source), (im_target, label_target, id_target)) in enumerate(
                iters
            ):
                label_source = label_source.to(self.device)
                label_target = label_target.to(self.device)
                im_source = im_source.to(self.device)
                im_target = im_target.to(self.device)
                xs_source = torch.tensor(source_sp_ds[id_source.numpy()][0]).to(self.device)
                xs_target = torch.tensor(target_sp_ds[id_target.numpy()][0]).to(self.device)
                y_source = torch.ones(len(im_source)).long().to(self.device)
                y_target = torch.zeros(len(im_target)).long().to(self.device)
                batch_coor = target_coor[id_target]

                z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)

                recon_source_c = self.decoder.dec[0](z_source, y_source)
                recon_target_c = self.decoder.dec[0](z_target, y_target)
                recon_source_s = self.decoder.dec[2](z_source, y_source)
                recon_target_s = self.decoder.dec[1](z_target, y_target)

                before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                before_lincls_feat_t, after_lincls_t = self.classifier(z_target)

                norm_feat_s = F.normalize(before_lincls_feat_s)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                # reconstruction loss
                recon_crit = nn.BCELoss()
                recon_loss = torch.tensor(0.0).to(self.device)

                recon_loss += recon_crit(recon_source_c, im_source) + recon_crit(recon_target_c, im_target)
                recon_loss += 0.5 * (recon_crit(recon_source_s, xs_source) + recon_crit(recon_target_s, xs_target))
                kl_loss = torch.tensor(0.0).to(self.device)
                kl_loss += kl_div(mu_source, var_source) + kl_div(mu_target, var_target)

                # classification loss
                criterion = CrossEntropyLabelSmooth(self.class_num, device=self.device, epsilon=0.1).to(self.device)
                cls_loss = torch.tensor(0.0).to(self.device)
                cls_loss = criterion(after_lincls_s, label_source)

                spa_loss = torch.tensor(0.0).to(self.device)
                ccd_loss = torch.tensor(0.0).to(self.device)

                if iteration > miditer1:
                    source_prototype = self.classifier.ProtoCLS.fc.weight
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                    _, _, Qt_all, wti_all = ubot_CCD(sim_tar_all, beta, device=self.device)
                    confidence_ot_all = (wti_all - torch.min(wti_all)) / (torch.max(wti_all) - torch.min(wti_all))
                    Escore_t, _, _, _ = ensemble_score(after_lincls_t, confidence_ot_all)
                    ###  calculate threshold
                    if threshold == None or iteration % (test_freq / 100) == 0:
                        with torch.no_grad():
                            idx_rand = torch.randperm(len(z_target))[:100]
                            ubot_feature_t_mix = mixup_target(z_target[idx_rand], device=self.device).to(self.device)

                            before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                            norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                            sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                            _, _, _, wti_mix = ubot_CCD(sim_tar_mix, beta_s, device=self.device)
                            confidence_ot_mix = (wti_mix - torch.min(wti_mix)) / (
                                torch.max(wti_mix) - torch.min(wti_mix)
                            )

                            Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                            threshold = torch.mean(Escore_mix_t)  # +torch.std(Escore_mix_t)
                            threshold = torch.min(
                                torch.tensor([threshold, torch.quantile(Escore_mix_t, (1 - resolution))])
                            )
                    high_conf_idx = Escore_t > threshold
                    id_target_high_conf = id_target[high_conf_idx]  # [Escore_t>threshold]
                    batch_coor_high_conf = batch_coor[high_conf_idx]  # [Escore_t>threshold]
                    # Spatial alignment loss
                    if iteration > miditer3:
                        # Spatial alignment loss
                        spa_dist_mat = distance_gmm(
                            mu_target[high_conf_idx],
                            mu_target[high_conf_idx],
                            var_target[high_conf_idx],
                            var_target[high_conf_idx],
                        )
                        spa_dist_mat = spa_dist_mat / torch.max(spa_dist_mat)
                        coor_dist_mat = pdists(batch_coor[high_conf_idx], squared=True)
                        coor_dist_mat = coor_dist_mat / torch.max(coor_dist_mat)
                        index = torch.topk(coor_dist_mat, k=k, largest=False)[1]
                        for i in range(len(id_target_high_conf)):
                            spa_loss += torch.norm(spa_dist_mat[i][index[i]] - coor_dist_mat[i][index[i]]) / (k - 1)
                        spa_loss /= len(id_target_high_conf)

                    # cell type ot alignment
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    #             sim = torch.matmul(norm_feat_t, source_prototype.t())
                    sim = torch.matmul(norm_feat_t[high_conf_idx], source_prototype.t())

                    # UOT-based CCD
                    pseudo_label_t_high_conf, new_beta, Q_t, wt_i = ubot_CCD(sim, beta, device=self.device)
                    pseudo_label_t_soft_high_conf = Q_t * (
                        (1 / torch.sum(Q_t, 1)).unsqueeze(1).expand(-1, Q_t.size()[1])
                    )
                    weight_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i))
                    # adaptive update for marginal probability vector
                    if True not in np.isnan(new_beta):
                        beta = mu * beta + (1 - mu) * new_beta
                    pred_label_t = torch.argmax(after_lincls_t, axis=1)
                    pred_label_t_soft = torch.softmax(after_lincls_t, axis=1)

                    if True not in torch.isnan(pseudo_label_t_soft_high_conf):
                        #                 ccd_loss = -torch.mean((pseudo_label_t_soft_high_conf * pred_label_t_soft[Escore_t > threshold_batch]).sum(-1))#
                        #                 ccd_loss += torch.mean(weight_t*(-pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft + 1e-10)).sum(dim=1))
                        ccd_loss += torch.mean(
                            weight_t
                            * (
                                -pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft[[high_conf_idx]] + 1e-10)
                            ).sum(dim=1)
                        )

                # neighbor loss
                local_loss = torch.tensor(0.0).to(self.device)
                spatial_loss = torch.tensor(0.0).to(self.device)
                neighbor_loss = torch.tensor(0.0).to(self.device)
                if iteration > miditer2:
                    dist_mat = pdists(batch_coor[high_conf_idx])
                    dist_mat += torch.max(dist_mat) * torch.eye(dist_mat.shape[0]).to(self.device)
                    _, spatial_nb_idx = torch.min(dist_mat, 1)
                    spatial_nb_output = after_lincls_t[high_conf_idx][spatial_nb_idx, :]
                    neighbor_Q_spatial = Q_t[spatial_nb_idx, :]
                    spatial_loss += -torch.sum(neighbor_Q_spatial * F.log_softmax(after_lincls_t[high_conf_idx]))
                    spatial_loss += -torch.sum(Q_t * F.log_softmax(spatial_nb_output))
                    spatial_loss /= 2 * len(batch_coor_high_conf)

                    #             feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                    feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                    mask = torch.eye(feat_mat.size(0), feat_mat.size(0)).bool().to(self.device)
                    feat_mat.masked_fill_(mask, -1 / temp)
                    local_nb_dist, local_nb_idx = torch.max(feat_mat, 1)
                    local_nb_output = after_lincls_t[high_conf_idx][local_nb_idx, :]
                    neighbor_Q_local = Q_t[local_nb_idx, :]
                    local_loss += -torch.sum(neighbor_Q_local * F.log_softmax(after_lincls_t[high_conf_idx]))
                    local_loss += -torch.sum(Q_t * F.log_softmax(local_nb_output))
                    local_loss /= 2 * len(batch_coor_high_conf)

                    neighbor_loss = (spatial_loss + local_loss) / 2

                # =====Total Loss=====
                loss = (
                    lambda_recon * recon_loss
                    + lambda_kl * kl_loss
                    + cls_loss
                    + lambda_cd * ccd_loss
                    + lambda_nb * neighbor_loss
                    + lambda_spa * spa_loss
                )

                optim_enc.zero_grad()
                optim_dec.zero_grad()
                optim_cls.zero_grad()
                loss.backward()
                optim_enc.step()
                optim_dec.step()
                optim_cls.step()

                self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

                iteration += 1
                if iteration % (test_freq / 10) == 0:
                    print(
                        "#Iter %d: Reconstruction loss: %f, KL loss: %f, CLS loss: %f, Spatial loss: %f, CCD loss: %f, Neighbor loss: %f"
                        % (
                            iteration,
                            recon_loss.item(),
                            kl_loss.item(),
                            cls_loss.item(),
                            spa_loss.item(),
                            ccd_loss.item(),
                            neighbor_loss.item(),
                        )
                    )

                if iteration % test_freq == 0 or iteration == maxiter:
                    with torch.no_grad():
                        source_prototype = self.classifier.ProtoCLS.fc.weight
                        source_z_bank = np.ones([len(source_sp_ds), self.feat_dim])
                        target_z_bank = np.ones([len(target_sp_ds), self.feat_dim])
                        Escore_bank = np.ones(len(target_sp_ds))
                        entropy_cls_bank = np.ones(len(target_sp_ds))
                        softmax_cls_bank = np.ones(len(target_sp_ds))
                        confidence_ot_bank = np.ones(len(target_sp_ds))
                        pred_label_bank_cls = np.ones(len(target_sp_ds))
                        pred_label_bank_ot = np.ones(len(target_sp_ds))
                        threshold_test = None

                        if beta is None:
                            source_prototype = self.classifier.ProtoCLS.fc.weight
                            beta = ot.unif(source_prototype.size()[0])

                        for _, ((im_source, label_source, id_source)) in enumerate(test_source_cm_dl):
                            label_source = label_source.to(self.device)
                            im_source = im_source.to(self.device)
                            xs_source = torch.tensor(source_sp_ds[id_source.numpy()][0]).to(self.device)
                            y_source = torch.ones(len(im_source)).long().to(self.device)
                            z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                            before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                            source_z_bank[id_source.cpu().numpy(), :] = before_lincls_feat_s.cpu().detach().numpy()
                            pseudo_label_s_cls = torch.argmax(after_lincls_s, axis=1)
                            norm_feat_s = F.normalize(before_lincls_feat_s)
                            sim_src = torch.matmul(norm_feat_s, source_prototype.t())
                            pseudo_label_s_ot, _, _, _ = ubot_CCD(sim_src, beta_s, device=self.device)

                        for _, ((im_target, label_target, id_target)) in enumerate(test_target_cm_dl):
                            im_target = im_target.to(self.device)
                            id_target = id_target.to(self.device)
                            xs_target = torch.tensor(target_sp_ds[id_target.cpu().numpy()][0]).to(self.device)
                            y_target = torch.zeros(len(im_target)).long().to(self.device)
                            z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)
                            before_lincls_feat_t, after_lincls_t = self.classifier(z_target)
                            target_z_bank[id_target.cpu().numpy(), :] = before_lincls_feat_t.cpu().detach().numpy()
                            pseudo_label_t_cls = y_target
                            pseudo_label_t_ot = y_target

                            pseudo_label_t_cls = torch.argmax(after_lincls_t, axis=1)

                            norm_feat_t = F.normalize(before_lincls_feat_t)
                            sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                            _, _, _, wt_i = ubot_CCD(sim_tar_all, beta, device=self.device)
                            confidence_ot_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i))

                            Escore_t, entropy_cls_t, softmax_cls_t, confidence_ot_t = ensemble_score(
                                after_lincls_t, confidence_ot_t
                            )

                            if threshold_test == None:
                                ubot_feature_t_mix = mixup_target(z_target, device=self.device).to(self.device)
                                before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                                norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                                sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                                _, _, _, wt_i_mix = ubot_CCD(sim_tar_mix, beta_s, device=self.device)
                                confidence_ot_mix = (wt_i_mix - torch.min(wt_i_mix)) / (
                                    torch.max(wt_i_mix) - torch.min(wt_i_mix)
                                )
                                Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                                threshold_test = torch.mean(Escore_mix_t)
                                threshold_test = torch.min(
                                    torch.tensor([threshold_test, torch.quantile(Escore_mix_t, (1 - resolution))])
                                )

                            pred_known_id = torch.nonzero(Escore_t >= threshold_test).view(-1)
                            pred_unknown_id = torch.nonzero(Escore_t < threshold_test).view(-1)
                            pseudo_label_t_cls[pred_unknown_id] = -1
                            pseudo_label_t_ot[pred_unknown_id] = -1
                            ##先分known和unknown 对known做ot 避免unknown的样本加进去不符合beta的分布
                            sim_tar_high_conf = torch.matmul(norm_feat_t[pred_known_id], source_prototype.t())
                            if sim_tar_high_conf.size()[0] > 0:
                                pseudo_label_t_high_conf, _, _, _ = ubot_CCD(
                                    sim_tar_high_conf, beta, device=self.device
                                )
                                pseudo_label_t_ot[pred_known_id] = pseudo_label_t_high_conf

                            idx_target = id_target.cpu().numpy()
                            pred_label_bank_cls[idx_target] = pseudo_label_t_cls.cpu().detach().numpy()
                            pred_label_bank_ot[idx_target] = pseudo_label_t_ot.cpu().detach().numpy()
                            Escore_bank[idx_target] = Escore_t.cpu().detach().numpy()
                            entropy_cls_bank[idx_target] = entropy_cls_t.cpu().detach().numpy()
                            softmax_cls_bank[idx_target] = softmax_cls_t.cpu().detach().numpy()
                            confidence_ot_bank[idx_target] = confidence_ot_t.cpu().detach().numpy()

                        if iteration == miditer1 and novel_cell_test:
                            dip_test = dip(Escore_bank, num_bins=100)
                            dip_p_value = dip_test[-1]
                            dip_test = dip_test[-2]
                            ensemble_total_scale = np.sqrt(Escore_bank)
                            BC = calculate_bimodality_coefficient(ensemble_total_scale)
                            print("bimodality of dip test:", dip_p_value, not dip_test)
                            print("bimodality coefficient:(>0.555 indicates bimodality)", BC, BC > 0.555)
                            bimodality = (not dip_test) or (BC > 0.555)
                            print("ood sample exists:", bimodality)
                            if bimodality:
                                resolution = 0.5
                            else:
                                resolution = 0.8

        adata_source = sc.AnnData(sp.csr_matrix(source_z_bank))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels.astype(int)]
        adata_source.obs["source"] = "scRNA"

        adata_target = sc.AnnData(sp.csr_matrix(target_z_bank))
        total_cell_types = list(cell_types) + ["Unknown"]
        adata_target.obs["pred_cls"] = [total_cell_types[i] for i in pred_label_bank_cls.astype(int)]
        adata_target.obs["pred_ot"] = [total_cell_types[i] for i in pred_label_bank_ot.astype(int)]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = Escore_bank
        adata_target.obs["source"] = "Spatial"

        return adata_source, adata_target, threshold_test

    def train_eval(
        self,
        source_cm_dl,
        target_cm_dl,
        source_sp_ds,
        target_sp_ds,
        spatial_coor,
        test_source_cm_dl,
        test_target_cm_dl,
        source_labels,
        target_labels,
        cell_types,
        common_cell_type,
        lr=2e-4,
        lambda_recon=2000,
        lambda_kl=0.5,
        lambda_spa=0.1,
        lambda_cd=0.001,
        lambda_nb=0.1,
        mu=0.6,
        temp=0.1,
        k=20,
        resolution=0.5,
        novel_cell_test=True,
        maxiter=6000,
        miditer1=2000,
        miditer2=5000,
        miditer3=4000,
        test_freq=1000,
    ):
        """
        Training and evaluating function for SPANN applying on test spatial dataset with ground truth labels

        :source_cm_dl: torch dataloader of the common genes scRNA-seq data
        :target_cm_dl: torch dataloader of the common genes spatial data
        :source_sp_ds: torch dataset of the scRNA-seq specific genes data
        :target_sp_ds: torch dataset of the spatial specific genes data
        :spatial_coor: pandas dataframe of the raw spatial coordinates, column names ['X','Y']
        :test_source_cm_dl: torch dataloader of the common genes scRNA-seq data for test or validate
        :test_target_cm_dl: torch dataloader of the common genes spatial data for test or validate
        :source_labels: integer labels for scRNA-seq data
        :target_labels: integer labels for spatial data
        :cell_types: list of all cell types exist in scRNA-seq or spatial datasets
        :common_cell_types: list of cell types exist in both scRNA-seq and spatial datasets

        :lr: learning rate for VAE and classifier, default=2e-4
        :lamnda_recon: training weight for the reconstruction loss, default=2000
        :lambda_kl: training weight for the KL-divergence loss, default=0.5
        :lambda_spa: training weight for the adjacency loss, default=0.1, we recomend to set it smaller when the spatial expression pattern is weak
        :lambda_cd: training weight for the unbalanced optimal transport alignment loss, default=0.001, we recomend to set it higher when the gap between scRNA-seq and spatial datasets is big
        :lambda_nb: training weight for the neighbor loss, default=0.1
        :mu: updating speed of beta by moving average, default=0.5
        :resolution: the expected minimum proportion of known cells, 0 means no constraints on novel cell discovery and 1 means no novel cells, default=0.5
        :novel_cell_test: whether to apply dip test and compute BC co-efficient to adaptively select resolution, default=True
        :maxiter: maximum iteration, default=6000
        :miditer1: after which iteration SPANN starts to conduct UOT alignment, default=2000
        :miditer2: after which iteration SPANN starts to train with neighbor loss, default=5000
        :miditer3: after which iteration SPANN starts to train with adjacency loss, default=4000
        :test_freq: test frequency, default=1000

        :return: two AnnData objects, adata_source and adata_target, containing the spatial coordinates, latent embeddings, E-scores, predictions and etc.

        """
        optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=5e-4)
        optim_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=5e-4)

        target_coor = torch.tensor(np.array(spatial_coor)).to(self.device)
        target_coor = torch.tensor(target_coor, dtype=torch.float)
        target_coor = target_coor / torch.max(target_coor, axis=0)[0]
        beta = None
        beta_s = ot.unif(self.class_num)
        threshold = None
        iteration = 0

        while iteration < maxiter:
            if len(source_sp_ds) > len(target_sp_ds):
                iters = zip(source_cm_dl, cycle(target_cm_dl))
            else:
                iters = zip(cycle(source_cm_dl), target_cm_dl)
            for minibatch_id, ((im_source, label_source, id_source), (im_target, label_target, id_target)) in enumerate(
                iters
            ):
                label_source = label_source.to(self.device)
                label_target = label_target.to(self.device)
                im_source = im_source.to(self.device)
                im_target = im_target.to(self.device)
                xs_source = torch.tensor(source_sp_ds[id_source.numpy()][0]).to(self.device)
                xs_target = torch.tensor(target_sp_ds[id_target.numpy()][0]).to(self.device)
                y_source = torch.ones(len(im_source)).long().to(self.device)
                y_target = torch.zeros(len(im_target)).long().to(self.device)
                batch_coor = target_coor[id_target]

                z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)

                recon_source_c = self.decoder.dec[0](z_source, y_source)
                recon_target_c = self.decoder.dec[0](z_target, y_target)
                recon_source_s = self.decoder.dec[2](z_source, y_source)
                recon_target_s = self.decoder.dec[1](z_target, y_target)

                before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                before_lincls_feat_t, after_lincls_t = self.classifier(z_target)

                norm_feat_s = F.normalize(before_lincls_feat_s)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                # reconstruction loss
                recon_crit = nn.BCELoss()
                recon_loss = torch.tensor(0.0).to(self.device)

                recon_loss += recon_crit(recon_source_c, im_source) + recon_crit(recon_target_c, im_target)
                recon_loss += 0.5 * (recon_crit(recon_source_s, xs_source) + recon_crit(recon_target_s, xs_target))
                kl_loss = torch.tensor(0.0).to(self.device)
                kl_loss += kl_div(mu_source, var_source) + kl_div(mu_target, var_target)

                # classification loss
                criterion = CrossEntropyLabelSmooth(self.class_num, device=self.device, epsilon=0.1).to(self.device)
                cls_loss = torch.tensor(0.0).to(self.device)
                cls_loss = criterion(after_lincls_s, label_source)

                spa_loss = torch.tensor(0.0).to(self.device)
                ccd_loss = torch.tensor(0.0).to(self.device)

                if iteration > miditer1:
                    source_prototype = self.classifier.ProtoCLS.fc.weight
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                    _, _, Qt_all, wti_all = ubot_CCD(sim_tar_all, beta, device=self.device)
                    confidence_ot_all = (wti_all - torch.min(wti_all)) / (torch.max(wti_all) - torch.min(wti_all))
                    Escore_t, _, _, _ = ensemble_score(after_lincls_t, confidence_ot_all)
                    ###  calculate threshold
                    if threshold == None or iteration % (test_freq / 100) == 0:
                        with torch.no_grad():
                            idx_rand = torch.randperm(len(z_target))[:100]
                            ubot_feature_t_mix = mixup_target(z_target[idx_rand], device=self.device).to(self.device)

                            before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                            norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                            sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                            _, _, _, wti_mix = ubot_CCD(sim_tar_mix, beta_s, device=self.device)
                            confidence_ot_mix = (wti_mix - torch.min(wti_mix)) / (
                                torch.max(wti_mix) - torch.min(wti_mix)
                            )

                            Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                            threshold = torch.mean(Escore_mix_t)  # +torch.std(Escore_mix_t)
                            threshold = torch.min(
                                torch.tensor([threshold, torch.quantile(Escore_mix_t, (1 - resolution))])
                            )
                    high_conf_idx = Escore_t > threshold
                    id_target_high_conf = id_target[high_conf_idx]  # [Escore_t>threshold]
                    batch_coor_high_conf = batch_coor[high_conf_idx]  # [Escore_t>threshold]
                    # Spatial alignment loss
                    if iteration > miditer3:
                        # Spatial alignment loss
                        spa_dist_mat = distance_gmm(
                            mu_target[high_conf_idx],
                            mu_target[high_conf_idx],
                            var_target[high_conf_idx],
                            var_target[high_conf_idx],
                        )
                        spa_dist_mat = spa_dist_mat / torch.max(spa_dist_mat)
                        coor_dist_mat = pdists(batch_coor[high_conf_idx], squared=True)
                        coor_dist_mat = coor_dist_mat / torch.max(coor_dist_mat)
                        index = torch.topk(coor_dist_mat, k=k, largest=False)[1]
                        for i in range(len(id_target_high_conf)):
                            spa_loss += torch.norm(spa_dist_mat[i][index[i]] - coor_dist_mat[i][index[i]]) / (k - 1)
                        spa_loss /= len(id_target_high_conf)

                    # cell type ot alignment
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])
                    #             sim = torch.matmul(norm_feat_t, source_prototype.t())
                    sim = torch.matmul(norm_feat_t[high_conf_idx], source_prototype.t())

                    # UOT-based CCD
                    pseudo_label_t_high_conf, new_beta, Q_t, wt_i = ubot_CCD(sim, beta, device=self.device)
                    pseudo_label_t_soft_high_conf = Q_t * (
                        (1 / torch.sum(Q_t, 1)).unsqueeze(1).expand(-1, Q_t.size()[1])
                    )
                    weight_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i))
                    # adaptive update for marginal probability vector
                    if True not in np.isnan(new_beta):
                        beta = mu * beta + (1 - mu) * new_beta
                    pred_label_t = torch.argmax(after_lincls_t, axis=1)
                    pred_label_t_soft = torch.softmax(after_lincls_t, axis=1)

                    if True not in torch.isnan(pseudo_label_t_soft_high_conf):
                        #                 ccd_loss = -torch.mean((pseudo_label_t_soft_high_conf * pred_label_t_soft[Escore_t > threshold_batch]).sum(-1))#
                        #                 ccd_loss += torch.mean(weight_t*(-pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft + 1e-10)).sum(dim=1))
                        ccd_loss += torch.mean(
                            weight_t
                            * (
                                -pseudo_label_t_soft_high_conf * torch.log(pred_label_t_soft[[high_conf_idx]] + 1e-10)
                            ).sum(dim=1)
                        )

                # neighbor loss
                local_loss = torch.tensor(0.0).to(self.device)
                spatial_loss = torch.tensor(0.0).to(self.device)
                neighbor_loss = torch.tensor(0.0).to(self.device)
                if iteration > miditer2:
                    dist_mat = pdists(batch_coor[high_conf_idx])
                    dist_mat += torch.max(dist_mat) * torch.eye(dist_mat.shape[0]).to(self.device)
                    _, spatial_nb_idx = torch.min(dist_mat, 1)
                    spatial_nb_output = after_lincls_t[high_conf_idx][spatial_nb_idx, :]
                    neighbor_Q_spatial = Q_t[spatial_nb_idx, :]
                    spatial_loss += -torch.sum(neighbor_Q_spatial * F.log_softmax(after_lincls_t[high_conf_idx]))
                    spatial_loss += -torch.sum(Q_t * F.log_softmax(spatial_nb_output))
                    spatial_loss /= 2 * len(batch_coor_high_conf)

                    #             feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                    feat_mat = torch.matmul(norm_feat_t[high_conf_idx], norm_feat_t[high_conf_idx].t()) / temp
                    mask = torch.eye(feat_mat.size(0), feat_mat.size(0)).bool().to(self.device)
                    feat_mat.masked_fill_(mask, -1 / temp)
                    local_nb_dist, local_nb_idx = torch.max(feat_mat, 1)
                    local_nb_output = after_lincls_t[high_conf_idx][local_nb_idx, :]
                    neighbor_Q_local = Q_t[local_nb_idx, :]
                    local_loss += -torch.sum(neighbor_Q_local * F.log_softmax(after_lincls_t[high_conf_idx]))
                    local_loss += -torch.sum(Q_t * F.log_softmax(local_nb_output))
                    local_loss /= 2 * len(batch_coor_high_conf)

                    neighbor_loss = (spatial_loss + local_loss) / 2

                # =====Total Loss=====
                loss = (
                    lambda_recon * recon_loss
                    + lambda_kl * kl_loss
                    + cls_loss
                    + lambda_cd * ccd_loss
                    + lambda_nb * neighbor_loss
                    + lambda_spa * spa_loss
                )

                optim_enc.zero_grad()
                optim_dec.zero_grad()
                optim_cls.zero_grad()
                loss.backward()
                optim_enc.step()
                optim_dec.step()
                optim_cls.step()

                self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

                iteration += 1
                if iteration % (test_freq / 10) == 0:
                    print(
                        "#Iter %d: Reconstruction loss: %f, KL loss: %f, CLS loss: %f, Spatial loss: %f, CCD loss: %f, Neighbor loss: %f"
                        % (
                            iteration,
                            recon_loss.item(),
                            kl_loss.item(),
                            cls_loss.item(),
                            spa_loss.item(),
                            ccd_loss.item(),
                            neighbor_loss.item(),
                        )
                    )

                if iteration % test_freq == 0 or iteration == maxiter:
                    with torch.no_grad():
                        source_corrcnts_ot = 0
                        source_corrcnts_cls = 0
                        source_cnts = 0
                        target_known_cnts = 0
                        target_known_corrcnts_ot = 0
                        target_known_corrcnts_cls = 0
                        target_unknown_cnts = 0
                        target_unknown_corrcnts_ot = 0
                        target_unknown_corrcnts_cls = 0
                        source_prototype = self.classifier.ProtoCLS.fc.weight
                        source_z_bank = np.ones([len(source_sp_ds), self.feat_dim])
                        target_z_bank = np.ones([len(target_sp_ds), self.feat_dim])
                        Escore_bank = np.ones(len(target_sp_ds))
                        entropy_cls_bank = np.ones(len(target_sp_ds))
                        softmax_cls_bank = np.ones(len(target_sp_ds))
                        confidence_ot_bank = np.ones(len(target_sp_ds))
                        pred_label_bank_cls = np.ones(len(target_sp_ds))
                        pred_label_bank_ot = np.ones(len(target_sp_ds))
                        threshold_test = None

                        if beta is None:
                            source_prototype = self.classifier.ProtoCLS.fc.weight
                            beta = ot.unif(source_prototype.size()[0])

                        for _, ((im_source, label_source, id_source)) in enumerate(test_source_cm_dl):
                            label_source = label_source.to(self.device)
                            im_source = im_source.to(self.device)
                            xs_source = torch.tensor(source_sp_ds[id_source.numpy()][0]).to(self.device)
                            y_source = torch.ones(len(im_source)).long().to(self.device)
                            z_source, mu_source, var_source = self.encoder(im_source, 0, y_source)
                            before_lincls_feat_s, after_lincls_s = self.classifier(z_source)
                            source_z_bank[id_source.cpu().numpy(), :] = before_lincls_feat_s.cpu().detach().numpy()
                            pseudo_label_s_cls = torch.argmax(after_lincls_s, axis=1)
                            norm_feat_s = F.normalize(before_lincls_feat_s)
                            sim_src = torch.matmul(norm_feat_s, source_prototype.t())
                            pseudo_label_s_ot, _, _, _ = ubot_CCD(sim_src, beta_s, device=self.device)

                            source_corrcnts_ot += torch.sum(pseudo_label_s_ot == label_source)
                            source_corrcnts_cls += torch.sum(pseudo_label_s_cls == label_source)
                            source_cnts += len(label_source)
                        for _, ((im_target, label_target, id_target)) in enumerate(test_target_cm_dl):
                            index_tf_known = [
                                label_target[i] in list(np.unique(source_labels)) for i in range(len(label_target))
                            ]
                            index_known = [i for i in range(len(index_tf_known)) if index_tf_known[i]]
                            index_unknown = [i for i in range(len(index_tf_known)) if not index_tf_known[i]]

                            im_target = im_target.to(self.device)
                            id_target = id_target.to(self.device)
                            xs_target = torch.tensor(target_sp_ds[id_target.cpu().numpy()][0]).to(self.device)
                            y_target = torch.zeros(len(im_target)).long().to(self.device)
                            z_target, mu_target, var_target = self.encoder(im_target, 0, y_target)
                            before_lincls_feat_t, after_lincls_t = self.classifier(z_target)
                            target_z_bank[id_target.cpu().numpy(), :] = before_lincls_feat_t.cpu().detach().numpy()
                            pseudo_label_t_cls = y_target
                            pseudo_label_t_ot = y_target

                            pseudo_label_t_cls = torch.argmax(after_lincls_t, axis=1)

                            norm_feat_t = F.normalize(before_lincls_feat_t)
                            sim_tar_all = torch.matmul(norm_feat_t, source_prototype.t())
                            _, _, _, wt_i = ubot_CCD(sim_tar_all, beta, device=self.device)
                            confidence_ot_t = (wt_i - torch.min(wt_i)) / (torch.max(wt_i) - torch.min(wt_i))

                            Escore_t, entropy_cls_t, softmax_cls_t, confidence_ot_t = ensemble_score(
                                after_lincls_t, confidence_ot_t
                            )

                            if threshold_test == None:
                                ubot_feature_t_mix = mixup_target(z_target, device=self.device).to(self.device)
                                before_lincls_feat_t_mix, logit_t_mix = self.classifier(ubot_feature_t_mix)
                                norm_feat_t_mix = F.normalize(before_lincls_feat_t_mix)
                                sim_tar_mix = torch.matmul(norm_feat_t_mix, source_prototype.t())
                                _, _, _, wt_i_mix = ubot_CCD(sim_tar_mix, beta_s, device=self.device)
                                confidence_ot_mix = (wt_i_mix - torch.min(wt_i_mix)) / (
                                    torch.max(wt_i_mix) - torch.min(wt_i_mix)
                                )
                                Escore_mix_t, _, _, _ = ensemble_score(logit_t_mix, confidence_ot_mix)
                                threshold_test = torch.mean(Escore_mix_t)
                                threshold_test = torch.min(
                                    torch.tensor([threshold_test, torch.quantile(Escore_mix_t, (1 - resolution))])
                                )

                            pred_known_id = torch.nonzero(Escore_t >= threshold_test).view(-1)
                            pred_unknown_id = torch.nonzero(Escore_t < threshold_test).view(-1)
                            pseudo_label_t_cls[pred_unknown_id] = -1
                            pseudo_label_t_ot[pred_unknown_id] = -1
                            ##先分known和unknown 对known做ot 避免unknown的样本加进去不符合beta的分布
                            sim_tar_high_conf = torch.matmul(norm_feat_t[pred_known_id], source_prototype.t())
                            if sim_tar_high_conf.size()[0] > 0:
                                pseudo_label_t_high_conf, _, _, _ = ubot_CCD(
                                    sim_tar_high_conf, beta, device=self.device
                                )
                                pseudo_label_t_ot[pred_known_id] = pseudo_label_t_high_conf

                            idx_target = id_target.cpu().numpy()
                            pred_label_bank_cls[idx_target] = pseudo_label_t_cls.cpu().detach().numpy()
                            pred_label_bank_ot[idx_target] = pseudo_label_t_ot.cpu().detach().numpy()
                            Escore_bank[idx_target] = Escore_t.cpu().detach().numpy()
                            entropy_cls_bank[idx_target] = entropy_cls_t.cpu().detach().numpy()
                            softmax_cls_bank[idx_target] = softmax_cls_t.cpu().detach().numpy()
                            confidence_ot_bank[idx_target] = confidence_ot_t.cpu().detach().numpy()
                            target_known_cnts += len(index_known)
                            target_unknown_cnts += len(index_unknown)

                        if iteration == miditer1 and novel_cell_test:
                            dip_test = dip(Escore_bank, num_bins=100)
                            dip_p_value = dip_test[-1]
                            dip_test = dip_test[-2]
                            ensemble_total_scale = np.sqrt(Escore_bank)
                            BC = calculate_bimodality_coefficient(ensemble_total_scale)
                            print("bimodality of dip test:", dip_p_value, not dip_test)
                            print("bimodality coefficient:(>0.45 indicates bimodality)", BC, BC > 0.45)
                            bimodality = (not dip_test) or (BC > 0.45)
                            print("ood sample exists:", bimodality)
                            if bimodality:
                                resolution = 0.5
                            else:
                                resolution = 0.8

                        index_tf_known_all = [
                            target_labels[i] in list(np.unique(source_labels)) for i in range(len(target_labels))
                        ]
                        index_known_all = [i for i in range(len(index_tf_known_all)) if index_tf_known_all[i]]
                        index_unknown_all = [i for i in range(len(index_tf_known_all)) if not index_tf_known_all[i]]
                        target_labels[index_unknown_all] = -1
                        Escore_known = Escore_bank[index_known_all]
                        Escore_unknown = Escore_bank[index_unknown_all]

                        entropy_cls_known = entropy_cls_bank[index_known_all]
                        entropy_cls_unknown = entropy_cls_bank[index_unknown_all]

                        softmax_cls_known = softmax_cls_bank[index_known_all]
                        softmax_cls_unknown = softmax_cls_bank[index_unknown_all]

                        confidence_ot_known = confidence_ot_bank[index_known_all]
                        confidence_ot_unknown = confidence_ot_bank[index_unknown_all]

                        target_known_cnts = len(index_known_all)
                        target_known_corrcnts_ot = np.sum(
                            pred_label_bank_ot[index_known_all] == target_labels[index_known_all]
                        )
                        target_known_corrcnts_cls = np.sum(
                            pred_label_bank_cls[index_known_all] == target_labels[index_known_all]
                        )
                        target_unknown_cnts = len(index_unknown_all)
                        target_unknown_corrcnts_ot = np.sum(
                            pred_label_bank_ot[index_unknown_all] == target_labels[index_unknown_all]
                        )
                        target_unknown_corrcnts_cls = np.sum(
                            pred_label_bank_cls[index_unknown_all] == target_labels[index_unknown_all]
                        )
                        acc_source_ot = source_corrcnts_ot / source_cnts
                        acc_source_cls = source_corrcnts_cls / source_cnts
                        acc_ot = (target_known_corrcnts_ot + target_unknown_corrcnts_ot) / (
                            target_known_cnts + target_unknown_cnts
                        )
                        acc_cls = (target_known_corrcnts_cls + target_unknown_corrcnts_cls) / (
                            target_known_cnts + target_unknown_cnts
                        )
                        acc_known_ot = target_known_corrcnts_ot / target_known_cnts
                        acc_unknown_ot = target_unknown_corrcnts_ot / target_unknown_cnts
                        acc_known_cls = target_known_corrcnts_cls / target_known_cnts
                        acc_unknown_cls = target_unknown_corrcnts_cls / target_unknown_cnts
                        print(
                            "#Iter %d: (OT) Source acc: %f, Target total acc : %f,  Target known acc: %f, Target unknown acc: %f"
                            % (iteration, acc_source_ot, acc_ot, acc_known_ot, acc_unknown_ot)
                        )
                        print(
                            "#Iter %d: (CLS) Source acc: %f, Target total acc : %f,  Target known acc: %f, Target unknown acc: %f"
                            % (iteration, acc_source_cls, acc_cls, acc_known_cls, acc_unknown_cls)
                        )

        adata_source = sc.AnnData(sp.csr_matrix(source_z_bank))
        adata_source.obs["cell_type"] = [cell_types[i] for i in source_labels.astype(int)]
        adata_source.obs["source"] = "scRNA"

        adata_target = sc.AnnData(sp.csr_matrix(target_z_bank))
        total_cell_types = list(common_cell_type) + ["Unknown"]
        adata_target.obs["cell_type"] = [cell_types[i] for i in target_labels.astype(int)]
        adata_target.obs["pred_cls"] = [total_cell_types[i] for i in pred_label_bank_cls.astype(int)]
        adata_target.obs["pred_ot"] = [total_cell_types[i] for i in pred_label_bank_ot.astype(int)]
        adata_target.obs[["X", "Y"]] = np.array(spatial_coor)
        adata_target.obs["Escore"] = Escore_bank
        adata_target.obs["source"] = "Spatial"

        return adata_source, adata_target, threshold_test
