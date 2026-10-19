import pickle, numpy as np, glob, sys
sys.path.insert(0, '.')
pool = pickle.load(open(glob.glob('/tmp/claude-0/ov/pool_elast_*.pkl')[0], 'rb'))
U = np.stack(pool['u']); UC = pool['uc_k2']; T = U - UC
tr, te = np.arange(32), np.arange(32, 40)
m = T[tr].mean(0)
rel = lambda p, i: np.linalg.norm(p - U[i]) / np.linalg.norm(U[i])
print("coarse only        rel-L2", np.mean([rel(UC[i], i) for i in te]))
print("coarse + MEAN corr rel-L2", np.mean([rel(UC[i] + m, i) for i in te]))
# how much does the solution itself vary between samples?
Um = U[tr].mean(0)
print("mean solution only rel-L2", np.mean([rel(Um, i) for i in te]))
print("sample-to-sample variation of correction / its norm:",
      np.mean([np.linalg.norm(T[i] - m) / np.linalg.norm(T[i]) for i in te]))
E = pool['E']; print("E field: min/max", np.min(E[0]), np.max(E[0]), " fraction stiff", np.mean(np.asarray(E[0]) > 1.5))
print("E differs between samples:", np.mean(np.asarray(E[0]) != np.asarray(E[1])))
