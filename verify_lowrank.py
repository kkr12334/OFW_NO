import numpy as np
exec(open('/tmp/claude-0/v/verify_channel.py').read().split('A1,_=')[0])
for w,c in [(4,1e4),(2,1e4),(4,1e2)]:
    A,_=q1_stiff(n,chan(c,w,3)); T,sizes=build(A,L); Ah=T.T@A.toarray()@T; s=0
    for i,k in enumerate(sizes):
        D=Ah[s:s+k,s:s+k]; s+=k
        if k<40: continue
        d=1/np.sqrt(np.diag(D)); Dj=D*d[:,None]*d[None,:]
        ev,V=np.linalg.eigh(Dj)
        r=[int((ev<ev[-1]/t).sum()) for t in (50,20,10)]
        # deflate the r small modes, kappa of remainder
        m=r[1]; kap=ev[-1]/ev[m] if m<k else np.nan
        print(f"w={w} c={c:5.0e} level-block {k:4d}: #eig < lmax/50,/20,/10 = {r}; kappa after removing {m} modes = {kap:.1f} (full {ev[-1]/ev[0]:.0f})")
