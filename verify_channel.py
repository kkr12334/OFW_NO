import numpy as np
exec(open('/tmp/claude-0/v/verify_gamblet.py').read().split('for kind,c in')[0])
def chan(c,w,seed):
    r=np.random.default_rng(seed); a=np.ones((n,n))
    for _ in range(3):
        i=r.integers(0,n-w); a[i:i+w,:]=c
        j=r.integers(0,n-w); a[:,j:j+w]=c
    return a
def js(M):
    d=1/np.sqrt(np.diag(M)); return np.linalg.cond(M*d[:,None]*d[None,:])
def blockjac(A,T,sizes):
    Ah=T.T@A.toarray()@T; Dj=np.zeros_like(Ah); s=0
    for k in sizes: Dj[s:s+k,s:s+k]=Ah[s:s+k,s:s+k]; s+=k
    import scipy.linalg as sla; ev=sla.eigh(Ah,Dj,eigvals_only=True); return ev[-1]/ev[0]
A1,_=q1_stiff(n,np.ones((n,n)))
for w in [1,2,4]:
  for c in [1e2,1e4]:
    A,_=q1_stiff(n,chan(c,w,3)); T,sizes=build(A,L); Ah=T.T@A.toarray()@T; s=0; o=[]
    for k in sizes: o.append(js(Ah[s:s+k,s:s+k])); s+=k
    Tf,_=build(A,L,Aref=A1)
    print(f"width {w} c={c:5.0e}: adapted Jacobi-kappa(D_k)={np.round(o[1:],1)}   frozen-T block-Jacobi kappa={blockjac(A,Tf,sizes):.1f}")
