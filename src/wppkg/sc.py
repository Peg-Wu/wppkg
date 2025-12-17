import logging
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Union
from tqdm.auto import tqdm
from scipy.sparse import issparse
from scipy.sparse import csc_matrix, csr_matrix
from .utils import get_sorted_indices_in_array_2d_by_row

logger = logging.getLogger(__name__)


def guess_is_lognorm(
    adata: ad.AnnData,
    epsilon: float = 1e-3,
    max_threshold: float = 15.0,
    validate: bool = True,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether there is a fractional component of values.
    Checks that data with decimal values is in expected log1p range.

    Args:
        adata: AnnData object to check
        epsilon: Threshold for detecting fractional values (default 1e-3)
        max_threshold: Maximum valid value for log1p normalized data (default 15.0)
        validate: Whether to validate the data is in valid log1p range (default True)

    Returns:
        bool: True if the input is lognorm, False if integer counts

    Raises:
        ValueError: If data has decimal values but falls outside
            valid log1p range (min < 0 or max >= max_threshold), indicating mixed or invalid scales
    """
    if adata.X is None:
        raise ValueError("adata.X is None")

    # Check for fractional values
    if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
        frac, _ = np.modf(adata.X.data)
    elif adata.isview:
        frac, _ = np.modf(adata.X.toarray())
    elif adata.X is None:
        raise ValueError("adata.X is None")
    else:
        frac, _ = np.modf(adata.X)  # type: ignore

    has_decimals = bool(np.any(frac > epsilon))

    if not has_decimals:
        # All integer values - assume raw counts
        logger.info("Data appears to be integer counts (no decimal values detected)")
        return False

    # Data has decimals - perform validation if requested
    # Validate it's in valid log1p range
    if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
        max_val = adata.X.max()
        min_val = adata.X.min()
    else:
        max_val = float(np.max(adata.X))
        min_val = float(np.min(adata.X))

    # Validate range
    if min_val < 0:
        raise ValueError(
            f"Invalid scale: min value {min_val:.2f} is negative. "
            f"Both Natural or Log1p normalized data must have all values >= 0."
        )

    if validate and max_val >= max_threshold:
        raise ValueError(
            f"Invalid scale: max value {max_val:.2f} exceeds log1p threshold of {max_threshold}. "
            f"Expected log1p normalized values in range [0, {max_threshold}), but found values suggesting "
            f"raw counts or incorrect normalization. Values above {max_threshold} indicate mixed scales "
            f"(some cells with raw counts, some with log1p values)."
        )

    # Valid log1p data
    logger.info(
        f"Data appears to be log1p normalized (decimals detected, range [{min_val:.2f}, {max_val:.2f}])"
    )

    return True


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[str, ad.AnnData]:
    """Split anndata on celltype column.

    Args:
        adata: AnnData object
        celltype_col: Column name in adata.obs that contains the celltype labels

    Returns:
        dict[str, AnnData]: Dictionary of AnnData objects, keyed by celltype
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }


def reverse_adata_to_raw_counts(
    adata: Union[str, ad.AnnData],
    int_tol: float = 1e-3
) -> ad.AnnData:
    if isinstance(adata, str):
        logger.info(f"Reading adata from {adata} ...")
        adata = sc.read_h5ad(adata)
    else:
        adata = adata.copy()

    if not guess_is_lognorm(adata):
        raise ValueError("Input adata is likely in raw (non-log1p) scale.")
    
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X = np.expm1(X)
    sorted_indices = get_sorted_indices_in_array_2d_by_row(
        arr=X,
        ignore_zero_values=True,
        descending=False,
        stable=None,
        n_jobs=1,
        enable_tqdm=False
    )
    # Get the minimum non-zero value for each cell.
    min_value = X[np.arange(X.shape[0]), [row[0] for row in sorted_indices]]

    logger.info("Start reversing back to raw counts.")
    res = []
    for cell_idx in tqdm(range(len(adata))):
        reverse_success = False
        for assume_min_value in range(1, 101):
            coef = min_value[cell_idx] / assume_min_value
            reversed_x = X[cell_idx] / coef
            round_reversed_X = np.round(reversed_x)
            is_near_integer = np.isclose(reversed_x, round_reversed_X, atol=int_tol)
            if np.all(is_near_integer):
                res.append(round_reversed_X)
                reverse_success = True
                break
        if not reverse_success:
            raise ValueError(
                f"Failed to reverse raw counts for cell {cell_idx}."
            )
    logger.info("Successfully reversed back to raw counts.")
    reversed_raw_counts_X = np.vstack(res).astype(np.float32)
    adata.X = csr_matrix(reversed_raw_counts_X)

    return adata
