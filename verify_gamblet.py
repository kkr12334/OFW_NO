import numpy as np, sys
sys.path.insert(0,'/tmp/claude-0/v')
exec(open('/tmp/claude-0/v/verify_layer.py').read().split('L=5; n=2**L')[0])
L=5; n=2**L; rng=np.random.default_rng(0)
from scipy.ndimage import gaussian_filter
def coef(kind,c):
    if kind=='lognormal':
        g=gaussian_filter(rng.standard_normal((n,n)),1.5); g=(g-g.min())/(g.max()-g.min()); return c**g
    a=np.ones((n,n))
    for r in rng.choice(n,4,replace=False): a[r,:]=c
    for q in rng.choice(n,3,replace=False): a[:,q]=c
    return a
def kappas_typeA(A):
    T,sizes=build(A,L); Ah=T.T@A.toarray()@T; out=[];s=0
    for k in sizes: out.append(np.linalg.cond(Ah[s:s+k,s:s+k])); s+=k
    return out
def kappas_gamblet(A):
    """Owhadi/BOD19-style: fine-to-coarse, test functions = nodal Q1 (refinable via C=P^T)."""
    Ak=A.toarray(); nk=2**L; out=[]
    while nk>2:
        nc=nk//2; P=prolong(nc).toarray(); C=P.T
        # kernel basis W (rows) of C: orthonormal complement of range(C^T)=range(P)
        Q,_=np.linalg.qr(P,mode='complete'); W=Q[:,P.shape[1]:].T
        B=W@Ak@W.T; out.append(np.linalg.cond(B))
        Cdag=np.linalg.solve(C@C.T,C)
        Cad=Cdag@(np.eye(Ak.shape[0])-Ak@W.T@np.linalg.solve(B,W))
        Ak=Cad@Ak@Cad.T; nk=nc
    return out[::-1]
for kind,c in [('lognormal',1e0),('lognormal',1e2),('lognormal',1e4),('channel',1e2),('channel',1e4)]:
    A,_=q1_stiff(n,coef(kind,c))
    print(f"{kind:9s} c={c:6.0e}  typeA(nodal coarse+lifting) kappa(D_k)={np.round(kappas_typeA(A)[1:],1)}   gamblet kappa(B_k)={np.round(kappas_gamblet(A),1)}")
print("-- Jacobi-scaled kappa of type-A D_k --")
def js(M):
    d=1/np.sqrt(np.diag(M)); return np.linalg.cond(M*d[:,None]*d[None,:])
for kind,c in [('lognormal',1e0),('lognormal',1e2),('lognormal',1e4),('channel',1e2),('channel',1e4)]:
    A,_=q1_stiff(n,coef(kind,c)); T,sizes=build(A,L); Ah=T.T@A.toarray()@T; s=0; o=[]
    for k in sizes: o.append(js(Ah[s:s+k,s:s+k])); s+=k
    print(f"{kind:9s} c={c:6.0e} Jacobi-scaled kappa(D_k)={np.round(o[1:],1)}")
