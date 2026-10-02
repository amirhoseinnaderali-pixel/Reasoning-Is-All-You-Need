# Result schema v2.0

Scientific results are immutable `result.json` artifacts validated by `cttr_vps/result_schema_v2.py` and described by `schemas/result.schema.json`.

The record captures experiment/run/task identity, seed, method, candidate-set hash, configuration/model/benchmark hashes, model calls, token usage, refinement/debug steps, visible evaluation, hidden evaluation, solved status, failure type, wall-clock time, budget usage, environment metadata, git SHA, execution mode, and status.

Results use exclusive file creation; an existing artifact cannot be silently overwritten.
