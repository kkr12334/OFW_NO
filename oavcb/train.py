"""Training: AdamW + one-cycle schedule on the mixed relative-L2 / relative-energy loss, resumable."""
import os
import time

import numpy as np
import torch

from .config import TrainConfig
from .pipeline import Problem, run_key, save_checkpoint


def rel_l2(p, t):
    return (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1)).mean()


def rel_energy(model, p, t):
    """mean_i ||p_i - t_i||_A / ||t_i||_A with A the sample's own stiffness (the batch currently set in `model`)."""
    num = model.energy_sq(p - t).clamp_min(0).sqrt()
    den = model.energy_sq(t).clamp_min(1e-30).sqrt()
    return (num / den).mean()


def loss_fn(model, p, t, kind='mix'):
    if kind == 'rel_l2':
        return rel_l2(p, t)
    en = rel_energy(model, p, t)
    return en if kind == 'energy' else 0.5 * (en + rel_l2(p, t))


def fit(prob: Problem, model, tcfg: TrainConfig, ckpt_path=None, key=None, log=print):
    """Train `model` on the training split of `prob`.  With OA-VCB (project='both') the loss is
    loss(projected output) + loss(plain cycle output).  Returns training seconds.  If ckpt_path exists the run resumes
    (or, for a finished checkpoint, is skipped)."""
    ds = prob.ds; n_tr = ds.n_tr
    steps = int(np.ceil(n_tr / tcfg.batch))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg.lr, total_steps=tcfg.epochs * steps)
    g = np.random.default_rng(tcfg.seed)
    start, t_prev = 0, 0.0
    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=prob.dev, weights_only=False)
        if key is not None and st.get('key') not in (None, key):
            raise SystemExit(f'checkpoint {ckpt_path} belongs to another configuration ({st.get("key")} vs {key})')
        model.load_state_dict(st['model'])
        if st.get('finished'):
            log(f'training already finished ({ckpt_path})'); return st.get('train_s', 0.0)
        opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch'])
        g.bit_generator.state = st['rng']; start, t_prev = st['epoch'], st['elapsed']
        log(f'resuming from epoch {start}')
    every = tcfg.ckpt_every or max(1, tcfg.epochs // 6)
    t0 = time.time() - t_prev
    for ep in range(start, tcfg.epochs):
        model.train(); tot = 0.0
        for idx in np.array_split(g.permutation(prob.ds.tr), steps):
            opt.zero_grad()
            out = prob.forward(model, idx)
            t = prob.T[idx]
            loss = loss_fn(model, out, t, tcfg.loss)
            if getattr(model, 'aux_out', None) is not None:               # OA-VCB: + loss of the plain cycle output
                loss = loss + loss_fn(model, model.aux_out.float(), t, tcfg.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
        if (ep + 1) % max(1, tcfg.epochs // 6) == 0 or ep == 0:
            log(f'  ep {ep + 1:>4}/{tcfg.epochs}  train {tcfg.loss} {tot / steps:.3e}  ({time.time() - t0:.0f}s)')
        if ckpt_path and (ep + 1) % every == 0 and ep + 1 < tcfg.epochs:
            save_checkpoint(ckpt_path, model, key, opt=opt.state_dict(), sch=sch.state_dict(), rng=g.bit_generator.state,
                            epoch=ep + 1, elapsed=time.time() - t0)
    train_s = time.time() - t0
    if ckpt_path:
        save_checkpoint(ckpt_path, model, key, finished=True, train_s=train_s)
    return train_s
