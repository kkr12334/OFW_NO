#!/bin/bash
# Amendment 2 chain (SHA-256 5480b6a9...): timing v2 (alone), then the reserve experiment (run once).
cd /home/claude/ofw
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=ext/runs
run() { name=$1; shift; echo "[$(date -u +%FT%TZ)] start $name" >> $R/chain_amendment2.log
        python3 -I "$@" > $R/$name.log 2>&1; ec=$?; echo "exit $ec" >> $R/$name.log
        echo "[$(date -u +%FT%TZ)] end $name exit $ec" >> $R/chain_amendment2.log; }
(cd ext && sha256sum -c STAGE1_AMENDMENT_2_SHA256.txt) >> $R/chain_amendment2.log 2>&1 || { echo "amendment 2 hash mismatch" >> $R/chain_amendment2.log; exit 1; }
run timing_v2_C_channels ext/stage1_timing_v2.py channels 50 $R/timing_v2_C_channels.json
run timing_v2_C_barriers ext/stage1_timing_v2.py barriers 50 $R/timing_v2_C_barriers.json
run reserve_channels ext/stage1_reserve.py channels
run reserve_barriers ext/stage1_reserve.py barriers
echo "[$(date -u +%FT%TZ)] chain done" >> $R/chain_amendment2.log
