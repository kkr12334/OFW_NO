#!/bin/bash
# Stage 1 confirmation chain (pre-confirmation note SHA-256 66be1dc3...).  Sequential: nothing else runs on the machine.
# Each computation runs once; exit status logged.  Started after the note was hashed.
cd /home/claude/ofw
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=ext/runs
run() { name=$1; shift; echo "[$(date -u +%FT%TZ)] start $name" >> $R/chain.log
        python3 -I "$@" > $R/$name.log 2>&1; ec=$?; echo "exit $ec" >> $R/$name.log
        echo "[$(date -u +%FT%TZ)] end $name exit $ec" >> $R/chain.log; }
(cd ext && sha256sum -c STAGE1_PRECONFIRMATION_SHA256.txt) >> $R/chain.log 2>&1 || { echo "pre-confirmation hash mismatch" >> $R/chain.log; exit 1; }
run confirm_C_channels ext/stage1_confirm.py C channels $R/confirm_C_channels.json
run confirm_C_barriers ext/stage1_confirm.py C barriers $R/confirm_C_barriers.json
run timing_C_channels ext/stage1_timing.py C channels 50 $R/timing_C_channels.json ext/dev/stage1_fitted_methods.json
run timing_C_barriers ext/stage1_timing.py C barriers 50 $R/timing_C_barriers.json ext/dev/stage1_fitted_methods.json
run d1_C_channels ext/stage1_d1.py C channels "resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt" 100 $R/d1_C_channels.json
run d1_C_barriers ext/stage1_d1.py C barriers "Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt" 100 $R/d1_C_barriers.json
run confirm_S_barriers ext/stage1_confirm.py S barriers $R/confirm_S_barriers.json
run confirm_LC_channels ext/stage1_confirm.py LC channels $R/confirm_LC_channels.json
for s in R2 R3; do for f in channels barriers; do run confirm_${s}_$f ext/stage1_confirm.py $s $f $R/confirm_${s}_$f.json; done; done
run d2_C_channels ext/stage1_d2.py C channels 100 $R/d2_C_channels.json
run d2_C_barriers ext/stage1_d2.py C barriers 100 $R/d2_C_barriers.json
echo "[$(date -u +%FT%TZ)] chain done" >> $R/chain.log
