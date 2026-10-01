# Result Schema v1.0

Each experiment writes result.json.

| Field | Meaning |
|---|---|
| experiment_id | Unique run identifier |
| problem_id | Benchmark problem identifier |
| method | Baseline or proposed method |
| models | Captured model identities |
| num_generation_attempts | Generation attempt count |
| num_refinement_rounds | Refinement count |
| candidate_count | Retained candidate count |
| compiled | Whether final candidate compiled |
| tests_passed/tests_failed | Supplied-test counts |
| solved | True only with at least one supplied test and all supplied tests passing |
| latency_seconds | Mean candidate execution latency, not end-to-end latency |
| model_calls | Observed/estimated provider calls |
| verification_attempts | Supplied tests run |
| tokens | Provider-reported tokens, otherwise null |
| error_type | Standardized failure category |
| wall_time_seconds | End-to-end orchestration time |
| environment/config/events | Runtime metadata, effective config, trace |

If samples are visible examples, solved means visible-sample correctness only. It must not be described as hidden/full-judge acceptance.