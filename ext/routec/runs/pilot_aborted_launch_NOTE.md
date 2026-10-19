# Aborted launch, 18 October 2026, 07:53

The first launch ran all four pilot runs as one background job. That job would have hit the 2-hour limit on a single
background task before the third run, so it was stopped about one minute after starting and each run is now launched
as its own background task.

When stopped, the run had passed its run-level gates and the input gates of geometry id 0 at contrast 1. It had computed
no outcome (operator status 'gates passed; outcomes pending'). Its record is kept unchanged as
`pilot_aborted_launch_channels_L3.json`. The relaunch uses the same seed, code and amendment, and regenerates the same
geometry deterministically.

# Interrupted run, 18 October 2026, about 09:05

The workspace container restarted while the barriers L3 pilot was running. By then the run had completed geometry id 0
at contrasts 1, 10² and 10⁴, and passed the gates for 10⁶. The process was lost. Following the rule in amendment §8
(repeat from the start with the same seed; keep the earlier record), its record and log are kept unchanged as
`pilot_interrupted_barriers_L3.json` and `.log`.

The run was restarted from the start with the same seed, code (hashes in `PILOT_CODE_SHA256_AT_LAUNCH.txt`, re-verified
after the restart) and amendment. The geometry is regenerated deterministically, so the three completed (id 0, contrast)
results must reproduce. The report checks this.
