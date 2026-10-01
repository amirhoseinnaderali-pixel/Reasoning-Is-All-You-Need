# Result Schema v1.0

Each run writes a machine-readable result.json.

Required fields:
- experiment_id
- problem_id
- method
- models
- candidate_count
- model_calls
- compiled
- tests_passed
- tests_failed
- total_tests
- solved
- latency_ms
- memory_mb
- verification_attempts
- error_type
- wall_time_seconds
- config
- events

The solved flag means only that the final candidate passed every supplied test and at least one supplied test existed. It must not be described as hidden/full-judge acceptance without a full judge.
