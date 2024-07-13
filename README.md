# Data Chunk Compaction in the Vectorized Execution [SIMD]

Current hash join can be faster using SIMD technique.
This repository contains code that we investigate the effect of SIMD on our proposed compacted vectorized hash join.

It implements several kinds of SIMD hash join, using the linear probing hash table or the seperated chaining hash table.

We do not publish the results from this repo, but we open source it for your interests.

Our conclusion is that:

This project is to investigate the effect of our proposed simd_compaction solution, i.e.
- SIMD cannot make the vectorized hash join faster.
- SIMD cannot make the proposed vectorized hash join faster.

The formal open-sourced code is provided in this repo [Chunk-Compaction-in-Vectorized-Execution](https://github.com/YimingQiao/Chunk-Compaction-in-Vectorized-Execution).

