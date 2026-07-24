# Local CV can be badly optimistic — measure variance, calibrate with LB

Observed: single 25% stratified holdout gave 0.9156 while the leaderboard gave
0.78095 (~0.13 gap). Small classes (3 samples) put 1 sample in validation and
show phantom 0.0 accuracy -> chasing those "failures" REGRESSED later iterations.
Rules: use stratified k-fold with mean±std; accept a new best only if gain is
significant vs fold variance AND improves most folds; submit early to calibrate
the local->LB offset; keep an untouched final fold.
