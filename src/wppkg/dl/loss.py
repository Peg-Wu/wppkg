import torch
import numpy as np
import scanpy as sc
from tqdm.auto import tqdm
from torch.distributions import NegativeBinomial


def nb_loss(x, mu, theta, scale_factor=1.0, eps=1e-6):
    """
    Negative‑binomial negative log‑likelihood.
        x, mu : [..., G]
        theta : [G] or [..., G]
    returns scalar
    """
    mu = mu * scale_factor
    logits = (mu + eps).log() - (theta + eps).log()  # NB parameterisation
    dist = NegativeBinomial(total_count=theta, logits=logits)
    return -dist.log_prob(x).mean()


def zinb_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, eps=1e-10):
    """
    Zero-Inflated Negative Binomial Loss.
        - mean: 
            - torch.clamp(torch.exp(...), min=1e-5, max=1e6)     [1e-5, 1e6]
            - gene expression proportions
        - disp: torch.clamp(F.softplus(...), min=1e-4, max=1e4)    [1e-4, 1e4]
            or torch.exp(torch.nn.Parameter(torch.randn(n_genes)))
        - pi: torch.sigmoid(...)                                   (0, 1)
    
    scale_factor:
        - situ 1:
            if normalized = raw x (1e4 / total_counts):
                scale_factor = total_counts / 1e4
        - situ 2:
            Model outputs gene expression proportions. 
                >>> px_scale_logits = model(...)
                >>> px_scale = F.softmax(px_scale_logits, dim=-1)
                >>> mean = px_scale * total_counts
            In this case, scale_factor = total_counts
    """
    mean = mean * scale_factor
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda * torch.square(pi)
        result += ridge

    result = torch.mean(result)
    return result


def get_wmse_weigths(
    adata_pert: sc.AnnData,
    groupby: str = "target_gene",
    method: str = "wilcoxon"  # t-test_overestim_var or wilcoxon
) -> dict[np.ndarray]:
    """
    NOTE:
        1. adata_pert should only contain perturbed cells and be log-normalized.
        2. adata_pert.var_names should contain the gene names.

    References:
        - wilcoxon: https://www.biorxiv.org/content/10.64898/2026.05.02.722054v1
        - t-test_overestim_var: https://arxiv.org/pdf/2506.22641
    """

    sc.tl.rank_genes_groups(
        adata=adata_pert,
        groupby=groupby,
        reference="rest",
        method=method
    )  # TODO: accelerate using pdex

    pert2weights = {}
    for pert in tqdm(adata_pert.obs[groupby].unique()):
        df = sc.get.rank_genes_groups_df(adata_pert, group=pert)
        df = df.set_index("names").loc[adata_pert.var_names]  # align the order of genes to adata_pert.var_names

        # step0: take the raw scores (e.g., t-statistic or z-score)
        scores = df["scores"].values
        # step1: absolute value
        scores = np.abs(scores)
        # step2: min-max normalization
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        # step3: amplify the effects by taking square
        scores = scores ** 2
        # step4: normalize the weights to sum up to 1
        scores = scores / (scores.sum() + 1e-8)

        pert2weights[pert] = scores

    return pert2weights