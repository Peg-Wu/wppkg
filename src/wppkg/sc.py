import logging
import numpy as np
import scanpy as sc
import anndata as ad
from enum import Enum
from tqdm.auto import tqdm
from scipy.sparse import issparse
from typing import Union, Optional
from scipy.sparse import csc_matrix, csr_matrix
from .utils import get_sorted_indices_in_array_2d_by_row

logger = logging.getLogger(__name__)


class DuplicatedFeatureHandling(Enum):
    mean_pooling = "mean_pooling"
    max_pooling = "max_pooling"


class UniformFeatureForAnnData:
    r"""
    Align an AnnData object to a predefined set of feature names (e.g., genes, ccres etc.) by reindexing and zero-padding missing features.

    This class ensures that the input AnnData is transformed to have exactly the same feature space as a provided target list.
    Features present in the target but missing in the input are filled with `zeros`; features not in the target are dropped.
    Duplicate feature names in the input are resolved via mean- or max-pooling before alignment.

    **Example:**

    Suppose you have an AnnData with 3 cells and genes ['A', 'B', 'C']:

    >>> import scanpy as sc
    >>> import numpy as np
    >>> from wppkg.sc import UniformFeatureForAnnData, DuplicatedFeatureHandling

    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])  # shape: (3, 3)
    >>> adata = sc.AnnData(X)
    >>> adata.var_names = ['A', 'B', 'C']

    >>> processor = UniformFeatureForAnnData(adata, duplicated_features_handling=DuplicatedFeatureHandling.max_pooling)
    >>> target_genes = ['B', 'C', 'D']  # Note: 'D' is not in original adata
    >>> aligned = processor(target_genes)

    >>> print(aligned.X)
    [[2. 3. 0.]
     [5. 6. 0.]
     [8. 9. 0.]]

    >>> print(aligned.var_names.tolist())
    ['B', 'C', 'D']

    Here, gene 'A' is dropped, 'B' and 'C' are kept in the new order, and 'D' (missing in input) is added as a zero column.
    """
    def __init__(
        self,
        input_h5ad: Union[str, ad.AnnData],
        feature_names_col: Optional[str] = None,
        duplicated_features_handling: DuplicatedFeatureHandling = DuplicatedFeatureHandling.max_pooling,
    ):
        r"""
        **Parameters:**

        input_h5ad : str or AnnData
            | Input data, either a path to an H5AD file or an in-memory AnnData object to be aligned.
        feature_names_col : str or None, optional (default: None)
            | Column name in `adata.var` to use as feature identifiers (e.g., `"gene_symbol"`, `"cCRE_id"`).
            | If `None`, the existing `adata.var_names` are used directly.
        duplicated_features_handling : DuplicatedFeatureHandling, optional (default: DuplicatedFeatureHandling.max_pooling)
            | Strategy for resolving duplicated feature names:
            | - `DuplicatedFeatureHandling.mean_pooling`: replace duplicates with their mean expression across cells.
            | - `DuplicatedFeatureHandling.max_pooling`: retain the duplicate showing the highest average expression.
        """
        # Read the input adata
        if isinstance(input_h5ad, str):
            logger.info(f"Read AnnData from {input_h5ad} ...")
            self.adata = sc.read_h5ad(input_h5ad)
        else:
            self.adata = input_h5ad.copy()

        logger.info(f"Input adata shape: {self.adata.shape}")

        # Convert adata.X to dense matrix
        if hasattr(self.adata.X, "toarray"):
            logger.info("Converting adata.X to dense matrix ...")
            self.adata.X = self.adata.X.toarray()

        if feature_names_col is not None:
            logger.info(f"Set adata.var_names from adata.var['{feature_names_col}'].")
            self.adata.var_names = self.adata.var[feature_names_col].astype(str)
        else:
            logger.warning("Using adata.var_names as feature names.")
        
        # Preprocess duplicated features if needed
        self.duplicated_features_handling = duplicated_features_handling
        # Ensure float for mean_pooling
        if np.issubdtype(self.adata.X.dtype, np.integer) and self.duplicated_features_handling == DuplicatedFeatureHandling.mean_pooling:
            logger.warning("adata.X uses integer dtype — converting to float32 to prevent silent truncation in mean pooling.")
            self.adata.X = self.adata.X.astype(np.float32)
        self._preprocess_duplicated_features()

        logger.info("Input AnnData prepared.")

        # Placeholder for the output adata
        self.output_adata = None

    def _preprocess_duplicated_features(self):
        unique_features, counts = np.unique(self.adata.var_names, return_counts=True)
        if not all(counts == 1):
            duplicated_features = [unique_features[idx] for idx in np.where(counts > 1)[0]]
            logger.info(f"Found {len(duplicated_features)} duplicated features ({duplicated_features}).")

            duplicated_features_X = []
            for feature in duplicated_features:
                if self.duplicated_features_handling == DuplicatedFeatureHandling.mean_pooling:
                    # NOTE: Mean-pooling for each duplicated feature
                    feature_X = self.adata[:, self.adata.var_names == feature].X.mean(axis=1).reshape(-1, 1)
                elif self.duplicated_features_handling == DuplicatedFeatureHandling.max_pooling:
                    # NOTE: Max-pooling for each duplicated feature
                    feature_X = self.adata[:, self.adata.var_names == feature].X
                    max_index = np.argmax(feature_X.mean(axis=0))
                    feature_X = feature_X[:, max_index].reshape(-1, 1)
                else:
                    raise ValueError(
                        f"Unknown duplicated features handling method: {self.duplicated_features_handling}"
                    )
                duplicated_features_X.append(feature_X)
            self.adata.var_names_make_unique()  # feature, feature-1, feature-2, ...
            duplicated_features_indices = self.adata.var_names.get_indexer(duplicated_features)
            self.adata.X[:, duplicated_features_indices] = np.concatenate(duplicated_features_X, axis=1)
        else:
            logger.info("No duplicated features found.")
        
    def __call__(
        self,
        target_feature_names: list[str]
    ) -> ad.AnnData:
        r"""
        Align the internal AnnData to the given target feature names by reindexing and zero-padding.

        Features in `target_feature_names` that are missing from the input AnnData are filled with zeros.
        The order of features in the output exactly matches `target_feature_names`.

        **Parameters:**

        target_feature_names : list[str]
            | List of feature names (e.g., gene symbols) defining the target feature space.
            | The output AnnData will have these as its `var_names`, in this exact order.

        **Returns:**

        AnnData
            | A new AnnData object with shape `(n_cells, len(target_feature_names))`.
            | Its `X` matrix is dense, and missing features are filled with zeros.
            | This h5ad file contains only X and var_names; all other attributes will be cleared.
        """
        logger.info("Unifying feature names ... (This will take some time.)")

        adata_origin_feature_names = self.adata.var_names.tolist()
        logger.info(f"Number of overlapping features: {len(set(target_feature_names) & set(adata_origin_feature_names))}")
        logger.info(f"Number of features to add (all values set to zero): {len(set(target_feature_names) - set(adata_origin_feature_names))}")

        target_feature_names_indices = self.adata.var_names.get_indexer(target_feature_names)
        valid = target_feature_names_indices != -1
        X = np.zeros((self.adata.n_obs, len(target_feature_names)), dtype=self.adata.X.dtype)
        X[:, valid] = self.adata.X[:, target_feature_names_indices[valid]]
        output_adata = ad.AnnData(X=X)
        output_adata.var_names = target_feature_names

        self.output_adata = output_adata
        logger.info(f"Output AnnData has registered to `self.output_adata`, shape: {output_adata.shape}")

        return output_adata


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
    int_tol: float = 1e-3,
    return_scaling_factors: bool = False
) -> Union[ad.AnnData, tuple[ad.AnnData, list[float]]]:
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
    res, scaling_factors = [], []
    for cell_idx in tqdm(range(len(adata))):
        reverse_success = False
        for assume_min_value in range(1, 101):
            coef = min_value[cell_idx] / assume_min_value
            reversed_x = X[cell_idx] / coef
            round_reversed_X = np.round(reversed_x)
            is_near_integer = np.isclose(reversed_x, round_reversed_X, atol=int_tol)
            if np.all(is_near_integer):
                res.append(round_reversed_X)
                scaling_factors.append(coef)
                reverse_success = True
                break
        if not reverse_success:
            raise ValueError(
                f"Failed to reverse raw counts for cell {cell_idx}."
            )
    logger.info("Successfully reversed back to raw counts.")
    reversed_raw_counts_X = np.vstack(res).astype(np.float32)
    adata.X = csr_matrix(reversed_raw_counts_X)

    if return_scaling_factors:
        return adata, scaling_factors
    else:
        return adata
