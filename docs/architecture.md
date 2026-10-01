# CTTR-VPS Architecture

Research question: how does additional inference-time computation change the probability of producing an objectively correct executable solution?

The repository's hardened pipeline is:

problem preprocessing -> multi-model planning -> candidate generation -> optimization -> objective execution-based candidate selection -> iterative execution-based debugging.

The experiment harness adds explicit methods:
- single-pass
- multi-sample
- self-refinement
- execution-based refinement
- CTTR-VPS

The primary correctness signal is objective execution, not model self-reported quality.

Important limitation: the supplied tests in ioi_multi_view.json are generally visible samples. Passing them is not equivalent to hidden/full-judge acceptance.

The harness writes a machine-readable result.json containing candidate count, tests passed/failed, solved flag, model calls, wall time, and error category.
