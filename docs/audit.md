# Scientific audit

Run `python scripts/scientific_audit.py`.

The audit emits `audit/scientific_audit.json` with machine-readable PASS/FAIL checks for benchmark freezing, model/runtime freezing, hidden-test isolation, immutable results, and real-mode preflight.

A FAIL caused by the missing external benchmark materialization is intentional fail-closed behavior; it must not be bypassed with a substitute dataset.
