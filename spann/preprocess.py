from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MaxAbsScaler, maxabs_scale
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def batch_scale(adata, use_rep="X", chunk_size=20000):
    for b in adata.obs["source"].unique():
        idx = np.where(adata.obs["source"] == b)[0]
        if use_rep == "X":
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
                )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.obsm[use_rep][idx])
            for i in range(len(idx) // chunk_size + 1):
                adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                    adata.obsm[use_rep][idx[i * chunk_size : (i + 1) * chunk_size]]
                )


def anndata_preprocess(adata_spa, adata_rna, highly_variable=2000, spatial_labels=False):
    """
    Preprocess the rna and spatial Anndata

    :adata_spa: AnnData file of spatial dataset, .obs contains 'X','Y', 'source'
    :adata_rna: AnnData file of rna dataset, .obs contains 'cell_type', 'source'
    :highly_variable: number of highly variable genes
    :spatial_labels: if there are ground truth spatial data labels, if True, adata_spa.obs should contains 'cell_type'
    :return: preprocessed AnnData file, adata_cm, adata_rna, adata_spa

    """
    adata_cm = adata_spa.concatenate(adata_rna, join="inner", batch_key="domain_id")

    if spatial_labels:
        # construct labels
        labels = np.zeros(len(adata_cm))
        cell_types = np.unique(adata_cm.obs["cell_type"])
        common_cell_type = [
            i for i in np.unique(adata_rna.obs["cell_type"]) if i in np.unique(adata_spa.obs["cell_type"])
        ]
        novel_cell_type = [
            i for i in np.unique(adata_spa.obs["cell_type"]) if i not in np.unique(adata_rna.obs["cell_type"])
        ]
        type_nums_common = np.arange(len(common_cell_type))
        type_nums_novel = np.arange(len(novel_cell_type)) + len(common_cell_type)
        adata_cm.obs["label"] = adata_cm.obs["cell_type"]
        for i in range(len(common_cell_type)):
            labels[adata_cm.obs["label"] == common_cell_type[i]] = type_nums_common[i]
        for i in range(len(novel_cell_type)):
            labels[adata_cm.obs["label"] == novel_cell_type[i]] = type_nums_novel[i]
        labels = np.asarray(labels, dtype=int)
        adata_cm.obs["labels"] = labels
        rna_labels = labels[adata_cm.obs["domain_id"] == "1"]
        spatial_labels = labels[adata_cm.obs["domain_id"] == "0"]
        adata_rna.obs["labels"] = pd.Categorical(rna_labels, categories=np.unique(rna_labels))
        adata_spa.obs["labels"] = pd.Categorical(spatial_labels, categories=np.unique(spatial_labels))
        print("rna_labels:", np.unique(rna_labels))
        print("spatial_labels:", np.unique(spatial_labels))
    else:
        labels = np.zeros(len(adata_cm))
        cell_types = np.unique(adata_rna.obs["cell_type"])
        type_nums = np.arange(len(cell_types))
        adata_cm.obs["label"] = adata_cm.obs["cell_type"]
        for i in range(len(cell_types)):
            labels[adata_cm.obs["label"] == cell_types[i]] = type_nums[i]
        labels = np.asarray(labels, dtype=int)
        adata_cm.obs["labels"] = labels
        adata_cm.obs["labels"][adata_cm.obs["domain_id"] == "1"] = -1
        rna_labels = labels[adata_cm.obs["domain_id"] == "1"]
        adata_rna.obs["labels"] = pd.Categorical(rna_labels, categories=np.unique(rna_labels))
        adata_spa.obs["labels"] = -1
        print("rna_labels:", np.unique(rna_labels))

    sc.pp.normalize_total(adata_cm)
    sc.pp.log1p(adata_cm)
    sc.pp.highly_variable_genes(
        adata_cm, n_top_genes=highly_variable, batch_key="domain_id", inplace=False, subset=True
    )
    batch_scale(adata_cm)

    sc.pp.normalize_total(adata_spa)
    sc.pp.log1p(adata_spa)
    sc.pp.highly_variable_genes(adata_spa, n_top_genes=highly_variable, inplace=False, subset=True)
    batch_scale(adata_spa)

    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=highly_variable, inplace=False, subset=True)
    batch_scale(adata_rna)

    return adata_cm, adata_spa, adata_rna


class SingleCellDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.shape = data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].toarray().squeeze()
        labels = self.labels[idx].squeeze()

        return x, labels, idx


def generate_dataloaders(adata_cm, adata_spa, adata_rna, batch_size=256):
    """
    Generate torch datasets and torch dataloaders from preprocessed AnnData files

    :adata_cm: AnnData file of the preprocessed common gene integrated scRNA-seq & spatial data
    :adata_spa: AnnData file of the preprocessed spatial data
    :adata_rna: AnnData file of the preprocessed scRNA-seq data
    :batch_size: batch size of the dataloaders, default=256
    :return: domain specific genes datasets - source_sp_ds,target_sp_ds, train dataloaders - source_cm_dl,target_cm_dl, test dataloaders - test_source_cm_dl,test_target_cm_dl
    """
    adata_cm_rna = adata_cm[adata_cm.obs["domain_id"] == "1"]
    adata_cm_spa = adata_cm[adata_cm.obs["domain_id"] == "0"]
    rna_labels = np.array(adata_rna.obs["labels"])
    spa_labels = np.array(adata_spa.obs["labels"])
    source_cm_ds = SingleCellDataset(adata_cm_rna.X, rna_labels)
    target_cm_ds = SingleCellDataset(adata_cm_spa.X, spa_labels)
    source_sp_ds = SingleCellDataset(adata_rna.X, rna_labels)
    target_sp_ds = SingleCellDataset(adata_spa.X, spa_labels)

    classes = rna_labels
    freq = Counter(classes)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in classes]
    sampler = WeightedRandomSampler(source_weights, len(source_cm_ds.labels))

    source_cm_dl = DataLoader(
        dataset=source_cm_ds, batch_size=batch_size, sampler=sampler, num_workers=4, drop_last=True
    )
    target_cm_dl = DataLoader(dataset=target_cm_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_source_cm_dl = DataLoader(
        dataset=source_cm_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
    )
    test_target_cm_dl = DataLoader(
        dataset=target_cm_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
    )
    return source_sp_ds, target_sp_ds, source_cm_dl, target_cm_dl, test_source_cm_dl, test_target_cm_dl


# output: x, domain_id, idx


def generate_ae_params(adata_cm, adata_spa, adata_rna, feat_dim=16):
    """
    Generate default autoencoder parameters from preprocessed spatial and rna data

    :adata_cm: AnnData file of the preprocessed common gene integrated scRNA-seq & spatial data
    :adata_spa: AnnData file of the preprocessed spatial data
    :adata_rna: AnnData file of the preprocessed scRNA-seq data
    :feat_dim: dimension of latent features, default=16
    :return: encoder_parameters, decoder_parameters, output dimensions, latent dimension
    """
    adatas = [adata_spa, adata_rna]
    num_cell = []
    num_gene = []
    for i, adata in enumerate(adatas):
        num_cell.append(adata.X.shape[0])
        num_gene.append(adata.X.shape[1])

    n_domain = len(adatas)
    enc = [["fc", 1024, 1, "relu"], ["fc", feat_dim, "", ""]]
    num_gene.append(adata_cm.X.shape[1])
    dec = {}
    dec[0] = [["fc", num_gene[n_domain], n_domain, "sigmoid"]]  # common decoder
    for i in range(1, n_domain + 1):
        dec[i] = [["fc", num_gene[i - 1], 1, "sigmoid"]]  # dataset-specific decoder

    x_dim = {}
    for key in dec.keys():
        x_dim[key] = dec[key][-1][1]
        z_dim = enc[-1][1]
    return enc, dec, x_dim, z_dim
