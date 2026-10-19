"""Independent param audit: build the module tree by hand from the class
definitions in untitled9.py and compare against the analytic formulas."""
import sys; sys.path.insert(0, '.')
from ofwno_benchmark_v2 import ofwno_param_count, fno_params, wno_params, mlp_params, match_capacity

def lin(i,o,bias=True): return i*o + (o if bias else 0)
def conv1x1(i,o): return i*o + o          # Conv2d(i,o,kernel_size=1)

def ofwno_ref(dims, mu=16, hidden=64, proj=16, linear_skip=True):
    t = lin(mu,hidden) + lin(hidden,hidden)           # mu_encoder
    cum = 0
    for out in dims:
        pd = proj if cum > 0 else 0
        if cum > 0: t += lin(cum, proj)               # CrossLevelCoupling
        t += lin(hidden+pd, hidden) + lin(hidden,hidden) + lin(hidden,out)  # .net
        t += lin(mu, out)                             # .skip
        if linear_skip: t += lin(mu, out, bias=False) # linear_heads
        cum += out
    return t

def fno_ref(h, modes, layers, cin=3):
    t = conv1x1(cin,h)
    for _ in range(layers):
        t += 2*h*h*modes*modes          # complex weights -> real DOF
        t += conv1x1(h,h)               # skip
    t += conv1x1(h,h) + conv1x1(h,1)    # output_proj
    return t

def wno_ref(h, layers, J=2, cin=3):
    t = conv1x1(cin,h)
    for _ in range(layers):
        t += conv1x1(h,h)                        # w_low
        t += J*conv1x1(3*h,3*h)                  # w_high
        t += conv1x1(h,h)                        # skip
    t += conv1x1(h,h) + conv1x1(h,1)
    return t

def mlp_ref(mu,out,dims):
    t,d = 0,mu
    for x in dims: t += lin(d,x); d=x
    return t + lin(d,out)

# level 6 channelized problem: interior-node counts per level
DIMS = [3, 5, 21, 105, 465, 1953, 1673]
ok = True
for name, a, b in [
    ("OFW-NO", ofwno_param_count(DIMS), ofwno_ref(DIMS)),
    ("FNO   ", fno_params(21,12,4),     fno_ref(21,12,4)),
    ("WNO   ", wno_params(80,4),        wno_ref(80,4)),
    ("MLP   ", mlp_params(16,sum(DIMS),(256,256,97)), mlp_ref(16,sum(DIMS),(256,256,97))),
]:
    flag = "OK " if a==b else "MISMATCH"
    ok &= (a==b)
    print(f"{name}  formula={a:>10,}  module-tree={b:>10,}  {flag}")

print("\n-- dict vs list input --")
d = {i:v for i,v in enumerate(DIMS)}
print("list:", ofwno_param_count(DIMS), " dict:", ofwno_param_count(d),
      "OK" if ofwno_param_count(DIMS)==ofwno_param_count(d) else "MISMATCH")

print("\n-- match_capacity on the dict build_pool actually returns --")
cfg = match_capacity(d)
print("\nall consistent:", ok)
