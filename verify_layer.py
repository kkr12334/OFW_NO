"""Verify algebraic claims behind the OA-lifting wavelet layer (Q1, unit square, Dirichlet)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
rng = np.random.default_rng(0)

def q1_stiff(n, acell):
    """n cells per side, acell (n,n) piecewise-constant coef. Returns interior-node stiffness."""
    Ke = np.array([[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]])/6.0
    N = n+1; idx = lambda i,j: i*N+j
    I=[];J=[];V=[]
    for i in range(n):
        for j in range(n):
            loc=[idx(i,j),idx(i,j+1),idx(i+1,j+1),idx(i+1,j)]
            for a in range(4):
                for b in range(4):
                    I.append(loc[a]);J.append(loc[b]);V.append(acell[i,j]*Ke[a,b])
    A = sp.csr_matrix((V,(I,J)),shape=(N*N,N*N))
    interior = np.array([idx(i,j) for i in range(1,n) for j in range(1,n)])
    return A[interior][:,interior].tocsc(), interior

def prolong_1d(nc):
    nf=2*nc; P=np.zeros((nf+1,nc+1))
    for i in range(nc+1): P[2*i,i]=1
    for i in range(nc): P[2*i+1,i]=.5; P[2*i+1,i+1]=.5
    return P[1:-1,1:-1]        # interior (Dirichlet)
def prolong(nc):
    p=prolong_1d(nc); return sp.csr_matrix(np.kron(p,p))
def new_nodes(nc):
    nf=2*nc; m=nf-1
    ii,jj=np.meshgrid(np.arange(1,nf),np.arange(1,nf),indexing='ij')
    new = ~((ii%2==0)&(jj%2==0))
    return np.flatnonzero(new.ravel()), m

def build(A, L, adapt=True, Aref=None):
    """Return list of level blocks and global synthesis matrix T (columns = basis funcs in fine nodal coords).
    If adapt=False, the lifting coefficients come from Aref (frozen transform) and are applied to A."""
    Acur = A.toarray(); Rcur = (Aref.toarray() if Aref is not None else Acur)
    n = 2**L; up = np.eye(Acur.shape[0]); cols=[]; blocks=[]
    for lev in range(L,1,-1):
        nc=n//2; P=prolong(nc).toarray(); nn,m=new_nodes(nc)
        E=np.zeros((m*m,len(nn))); E[nn,np.arange(len(nn))]=1
        Rc=P.T@Rcur@P
        C=-np.linalg.solve(Rc, P.T@Rcur@E)          # update / lifting step
        Psi=E+P@C
        cols.append((lev, up@Psi))
        Rcur=Rc; Acur=P.T@Acur@P; up=up@P; n=nc
    cols.append((1, up))                               # coarsest scaling functions
    cols=cols[::-1]
    T=np.hstack([c for _,c in cols]); sizes=[c.shape[1] for _,c in cols]
    return T, sizes

def blocks_offdiag(M, sizes):
    off=M.copy(); s=0
    for k in sizes: off[s:s+k,s:s+k]=0; s+=k
    return np.linalg.norm(off)/np.linalg.norm(M)

L=5; n=2**L
def coef(kind, contrast):
    if kind=='const': return np.ones((n,n))
    if kind=='lognormal':
        g=rng.standard_normal((n,n)); from scipy.ndimage import gaussian_filter
        g=gaussian_filter(g,1.5); g=(g-g.min())/(g.max()-g.min())
        return contrast**g
    if kind=='channel':
        a=np.ones((n,n))
        for r in rng.choice(n,4,replace=False): a[r,:]=contrast
        for c in rng.choice(n,3,replace=False): a[:,c]=contrast
        return a

print("== 1. exactness of adapted transform (block-diagonal, Green's form, inverse) ==")
for kind,cst in [('const',1),('lognormal',1e2),('lognormal',1e4),('channel',1e4)]:
    A,_=q1_stiff(n,coef(kind,cst)); T,sizes=build(A,L)
    Ah=T.T@A.toarray()@T
    Dinv=np.zeros_like(Ah); s=0
    for k in sizes: Dinv[s:s+k,s:s+k]=np.linalg.inv(Ah[s:s+k,s:s+k]); s+=k
    G=T@Dinv@T.T; Ainv=np.linalg.inv(A.toarray())
    print(f"{kind:9s} c={cst:7.0e}  offdiag/||T^TAT||={blocks_offdiag(Ah,sizes):.1e}  "
          f"||T D^-1 T^T - A^-1||/||A^-1||={np.linalg.norm(G-Ainv)/np.linalg.norm(Ainv):.1e}  det(T)={np.linalg.det(T):.3f}"
          f"  kappa(D_k)={[round(np.linalg.cond(Ah[sum(sizes[:i]):sum(sizes[:i+1]),sum(sizes[:i]):sum(sizes[:i+1])]),1) for i in range(len(sizes))]}")

print("\n== 2. frozen transform (built for a=1) applied to A(a): residual cross-scale coupling ==")
A1,_=q1_stiff(n,np.ones((n,n)))
for kind,cst in [('lognormal',1e1),('lognormal',1e2),('lognormal',1e4),('channel',1e2),('channel',1e4)]:
    A,_=q1_stiff(n,coef(kind,cst)); T,sizes=build(A,L,Aref=A1)
    Ah=T.T@A.toarray()@T
    # strengthened C-S between coarsest+... : use block-Jacobi preconditioned condition number as coupling measure
    Dj=np.zeros_like(Ah); s=0
    for k in sizes: Dj[s:s+k,s:s+k]=Ah[s:s+k,s:s+k]; s+=k
    ev=sla.eigh(Ah,Dj,eigvals_only=True)
    Ainv=np.linalg.inv(A.toarray()); G=T@np.linalg.inv(Dj)@T.T
    print(f"{kind:9s} c={cst:7.0e}  offdiag={blocks_offdiag(Ah,sizes):.2f}  kappa(Dj^-1 Ah)={ev[-1]/ev[0]:8.1f}  "
          f"||T Dj^-1 T^T - A^-1||/||A^-1|| = {np.linalg.norm(G-Ainv)/np.linalg.norm(Ainv):.2f}")

print("\n== 3. scale self-similarity of D_k for a=1 (interior stencil, justifies weight sharing) ==")
T,sizes=build(A1,L); Ah=T.T@A1.toarray()@T
s=0
for i,k in enumerate(sizes):
    D=Ah[s:s+k,s:s+k]; s+=k
    if k<9: continue
    j=np.argmax(np.abs(np.diag(D))*0+ np.arange(k)==k//2)   # a central column
    col=np.sort(np.abs(D[:,j]))[::-1][:5]
    print(f"level block size {k:5d}: diag range [{np.diag(D).min():.3f},{np.diag(D).max():.3f}]  5 largest |entries| in central col {np.round(col,3)}")

print("\n== 4. decay of D_k^{-1} (adapted, lognormal 1e2), finest level ==")
A,_=q1_stiff(n,coef('lognormal',1e2)); T,sizes=build(A,L); Ah=T.T@A.toarray()@T
k=sizes[-1]; D=Ah[-k:,-k:]; Dinv=np.linalg.inv(D)
nn,m=new_nodes(n//2); xy=np.stack(np.unravel_index(nn,(m,m)),1)*(1.0/n)
c=np.argmin(((xy-0.5)**2).sum(1)); d=np.sqrt(((xy-xy[c])**2).sum(1))*n  # distance in fine h
for r in [0,1,2,4,8,12]:
    mask=(d>=r)&(d<r+1)
    if mask.any(): print(f"  dist {r:2d}h: max|Dinv|/|Dinv_cc| = {np.abs(Dinv[c,mask]).max()/abs(Dinv[c,c]):.2e}")
