import numpy as np, scipy.linalg as sla
exec(open('/tmp/claude-0/v/verify_channel.py').read().split('A1,_=')[0])
def build_local(A, rho):
    Acur=A.toarray(); nn_=2**L; up=np.eye(Acur.shape[0]); cols=[]
    while nn_>2:
        nc=nn_//2; P=prolong(nc).toarray(); nn,m=new_nodes(nc)
        E=np.zeros((m*m,len(nn))); E[nn,np.arange(len(nn))]=1
        Ac=P.T@Acur@P; Bm=P.T@Acur@E; mc=nc-1
        cxy=np.stack(np.unravel_index(np.arange(mc*mc),(mc,mc)),1)*2+1   # coarse node fine-index coords (interior offset)
        fxy=np.stack(np.unravel_index(nn,(m,m)),1)
        C=np.zeros((mc*mc,len(nn)))
        for t in range(len(nn)):
            S=np.flatnonzero(np.max(np.abs(cxy-fxy[t]),1)<=2*rho)
            if len(S): C[S,t]=-np.linalg.solve(Ac[np.ix_(S,S)],Bm[S,t])
        cols.append(up@(E+P@C)); Acur=Ac; up=up@P; nn_=nc
    cols.append(up); cols=cols[::-1]
    return np.hstack(cols),[c.shape[1] for c in cols]
def bj(A,T,sizes):
    Ah=T.T@A.toarray()@T; Dj=np.zeros_like(Ah); s=0
    for k in sizes: Dj[s:s+k,s:s+k]=Ah[s:s+k,s:s+k]; s+=k
    ev=sla.eigh(Ah,Dj,eigvals_only=True); return ev[-1]/ev[0]
for name,a in [('lognormal 1e4',None),('channel w4 1e2',chan(1e2,4,3)),('channel w4 1e4',chan(1e4,4,3)),('channel w2 1e4',chan(1e4,2,3))]:
    if a is None:
        from scipy.ndimage import gaussian_filter
        g=gaussian_filter(np.random.default_rng(1).standard_normal((n,n)),1.5); g=(g-g.min())/(g.max()-g.min()); a=1e4**g
    A,_=q1_stiff(n,a)
    print(name, ' '.join(f"rho={r}:{bj(A,*build_local(A,r)):.1f}" for r in [1,2,3,5]))
