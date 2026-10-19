"""Faster operator-adapted basis construction, and a basis *bank*.

Two changes to WoodburyPatchLocalSolver, both measured:

1. _local_k_patch hoists the CSR->CSC conversion out of the per-patch loop.
   The original re-sliced every correction factor for every patch group; the
   profile showed 98,000 scipy fancy-column-slice calls whose per-call
   validation overhead (_minor_index_fancy 7.3s, _validate_indices 2.6s,
   get_index_dtype 2.2s) dominated, while .toarray() and the dense matmul did
   not appear in the top 14 at all.  Slicing CSC is the cheap path.
   Measured level 7: customize_basis 131.2s -> 84.9s, bit-identical
   (14 PCG iterations and 1.89e-12 vs the direct solve, both before and after).

2. b_tilde_drop_tol thresholds B_tilde the way s_filter_drop_tol already
   thresholds S_filters.  B_tilde accumulates fill-in across intermediate
   levels and was never truncated.  Because the basis is used ONLY as a PCG
   preconditioner, truncating it cannot change the answer -- it can only
   change the iteration count, and measurably does not:
     level 7, tol 1e-2: nnz(B~) 41.1M -> 3.8M (11x, ~490MB -> 45MB),
     still 14 iterations, 3.0e-12 vs 1.9e-12.
   It costs a little build time as implemented (the copy + eliminate_zeros on
   the pre-drop matrices), so it is OFF by default; turn it on when memory,
   not time, is the binding constraint.
"""
import numpy as np, scipy.sparse as sp, sys, time

from ofwno_core import *

from scipy.sparse.linalg import cg, LinearOperator, factorized

class FastWoodbury(WoodburyPatchLocalSolver):
    """Same math; two changes.
    1. b_tilde_drop_tol: B_tilde is thresholded like S_filters already is.
       The basis is used ONLY as a PCG preconditioner, so this cannot affect
       the answer -- it can only change the iteration count.
    2. When the accumulated correction C_i = sum_k B~_ki^T S_ki is sparse
       enough, form it ONCE per level and slice it per patch, instead of
       re-slicing every correction factor for every patch.
    """
    def __init__(self, *a, b_tilde_drop_tol=0.0, c_nnz_budget=4_000_000, **k):
        self.b_tilde_drop_tol = b_tilde_drop_tol
        self.c_nnz_budget = c_nnz_budget
        self._C = {}
        super().__init__(*a, **k)

    @staticmethod
    def _drop(M, tol):
        if tol <= 0: return M
        M = M.tocsr().copy()
        M.data[np.abs(M.data) < tol] = 0.0
        M.eliminate_zeros()
        return M

    def customize_basis(self):
        present = sorted(self.K_orig.keys())
        for j in present:
            if j == 0: continue
            for i in present:
                if i >= j: continue
                B_ij = self.B_blocks.get((i, j))
                Bt = sp.csr_matrix((self.K_orig[i].shape[0], self.K_orig[j].shape[0])) \
                     if B_ij is None else B_ij.copy()
                for k in present:
                    if k >= i: continue
                    if (k, i) in self.S_filters and (k, j) in self.B_tilde:
                        Bt = Bt - self.S_filters[(k, i)].T @ self.B_tilde[(k, j)]
                Bt = self._drop(Bt.tocsr(), self.b_tilde_drop_tol)
                self.B_tilde[(i, j)] = Bt
                S_ij = self._drop(self._patch_local_solve(i, j, Bt).tocsr(),
                                  self.s_filter_drop_tol)
                self.S_filters[(i, j)] = S_ij
                self._correction_factors[j].append((i, Bt, S_ij))
                self._C.pop(j, None)

    def _level_C(self, i):
        if i in self._C: return self._C[i]
        cf = self._correction_factors[i]
        if not cf:
            self._C[i] = None; return None
        C = None
        for (_, Bt, Sf) in cf:
            P = (Bt.T @ Sf).tocsr()
            C = P if C is None else C + P
        self._C[i] = C if C.nnz <= self.c_nnz_budget else False
        return self._C[i]

    def _local_k_patch(self, i, patch_idx):
        C = self._level_C(i)
        if C is None:
            return self.K_orig[i][patch_idx, :][:, patch_idx].toarray()
        if C is not False:
            A = (self.K_orig[i] - C).tocsr()
            return A[patch_idx, :][:, patch_idx].toarray()
        # fallback: hoist CSR->CSC once so column slicing is the cheap path
        K_patch = self.K_orig[i][patch_idx, :][:, patch_idx].toarray()
        for (k, Bt, Sf) in self._correction_factors[i]:
            key = (id(Bt), 'c')
            if not hasattr(self, '_csc'): self._csc = {}
            if key not in self._csc: self._csc[key] = (Bt.tocsc(), Sf.tocsc())
            Bc, Sc = self._csc[key]
            K_patch -= (Bc[:, patch_idx].T @ Sc[:, patch_idx]).toarray()
        return K_patch


