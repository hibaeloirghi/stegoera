# raw_scores_7788093.csv is INVALID for watermarked vectors

Job 7788093 re-scored saved D0 carrier vectors with a freshly generated PRC key. `KeyGen` is not reproducible from
its seed (the one-time pad, test bits and generator matrix use `galois.GF.Random` with fresh entropy), so that key is
not the key D0 used. Unwatermarked-baseline rows are random-level under any key and are only a sanity check.
The file is kept for history and is not evidence for or against any embedding. D0's own CSV is valid (sender and
receiver share one process and one key). See AUDIT.md finding F.
