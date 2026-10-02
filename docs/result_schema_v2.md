# Result schema v2.0

The canonical scientific schema is `schemas/result.schema.json` and `cttr_vps/result_schema_v2.py`.

Required provenance includes experiment/run/task/seed identity, method, candidate-set hash, generated candidate count, visible-test hash, hidden-test hash, configuration/model/benchmark hashes, model revision, model calls, token usage, refinement/debug steps, visible and hidden evaluations, solved flag, failure type, wall-clock time, budget usage, environment metadata, git SHA, execution mode, and status.

Results are immutable: the writer uses exclusive file creation and refuses to overwrite an existing result artifact.

Visible-test success is never described as hidden/full-judge correctness.
