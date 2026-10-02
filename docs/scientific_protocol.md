# CTTR-VPS Scientific Protocol

The public benchmark materialization is now frozen from the legitimate HLCE IOI source. EXP-001 is **not** fully frozen yet because the independent hidden-test artifact is not available.

Visible tests are limited to the public sample tests from the upstream source. The public benchmark materialization contains no hidden-test data.

The hidden artifact is loaded only after final-candidate selection from `CTTR_HIDDEN_TESTS_PATH` in real mode. Real preflight requires every manifest task to have a matching hidden-test hash and a frozen hidden-test artifact.

The experiment must not start until the hidden artifact is supplied and the isolation tests/audit pass.
