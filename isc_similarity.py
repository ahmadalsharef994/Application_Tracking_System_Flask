from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity

def isc_similarity(X, Y=None, dense_output=True):
        """Compute cosine similarity between samples in X and Y.
        Cosine similarity, or the cosine kernel, computes similarity as the
        normalized dot product of X and Y:
            K(X, Y) = <X, Y> / (||X||*||Y||)
        On L2-normalized data, this function is equivalent to linear_kernel.
        Read more in the :ref:`User Guide <cosine_similarity>`.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.
        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features), \
                default=None
            Input data. If ``None``, the output will be the pairwise
            similarities between all samples in ``X``.
        dense_output : bool, default=True
            Whether to return dense output even when the input is sparse. If
            ``False``, the output is sparse if both input arrays are sparse.
            .. versionadded:: 0.17
               parameter ``dense_output`` for dense output.
        Returns
        -------
        kernel matrix : ndarray of shape (n_samples_X, n_samples_Y)
        """
        # to avoid recursive import

        X, Y = pairwise.check_pairwise_arrays(X, Y)

        X_normalized = pairwise.normalize(X, copy=True, norm='l1')
        if X is Y:
            Y_normalized = X_normalized
        else:
            Y_normalized = pairwise.normalize(Y, copy=True, norm='l1')

        K = pairwise.safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

        return K