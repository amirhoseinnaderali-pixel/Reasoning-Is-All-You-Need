# CTTR-VPS Architecture

The scientific pipeline is:

problem preprocessing
→ planning
→ candidate generation
→ optimization/deduplication
→ objective candidate selection using visible tests
→ execution-based debugging/refinement
→ final candidate
→ independent hidden evaluation

The repository contains explicit C0–C4 conditions. Visible tests are the only tests available before final-candidate selection. Hidden tests are loaded only after selection.

The real sandbox is a digest-pinned Docker execution path with network disabled, read-only root filesystem, dropped capabilities, no-new-privileges, CPU/memory/PID limits, controlled /tmp, and execution timeouts.

Validation mode performs software-only checks. Smoke mode uses the real model adapter and real sandbox but is always marked `VALIDATION_ONLY`. Real mode fails closed when any frozen scientific prerequisite is missing.
