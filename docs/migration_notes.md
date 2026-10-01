# Migration Notes
- Original top-level modules remain to preserve existing imports and historical workflows.
- The research harness lives under cttr_vps/ and scripts/.
- Historical planning JSON, multi-view data, and generated C++ artifacts are retained.
- Embedded credentials were removed from the current source tree and replaced by environment variables.
- Previously committed credentials should be rotated; history rewriting is not performed automatically.
- The GitHub repository slug has not been renamed because the available repository mutation interface does not expose a rename operation.
- External API experiments are not executed during structural migration.