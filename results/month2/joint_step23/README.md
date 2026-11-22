# Joint delivery, steps 2 and 3 (index for Opus and Neon)

Start from `package/INDEX.md` (generated on the cluster by `scripts/57_t_joint_package.py`, job 30083497, code SHA `682fe29496d9852c82af2a5d6fb59f05eb329006`; its cluster paths refer to the per-frame CSV that are not in git).

| What | Where in git |
| --- | --- |
| Joint index, tables, checks, references | `results/month2/joint_step23/package/` |
| Step 2 (Cursor), branch `sweep/c149-temperature-k`, SHA `be18ad007e66fbf2565ad3d5486c6513fc5c3e4c` | `results/month2/temperature_k_sweep/INDEX.md`, `docs/month2/11_temperature_k_sweep.md` (on Cursor's branch) |
| Step 3 v2 (assisted, 12 runs, three GLU continued) | `results/month2/temperature_assisted_c149/package_v2/INDEX.md`, `matrix_frozen_v2.json`, `chain_v2_submitted.json`, `extension/` |
| Step 3 v1 (1000-epoch state, unchanged) | `results/month2/temperature_assisted_c149/package/` (published at `aa8f65f`) |
| Reports | `docs/month2/12_temperature_input_study.md` (sections 9 and 10), `docs/month2/temperature_study_handoff.md` |
| Hashes of Cursor's published files | `results/month2/joint_step23/cursor_published_be18ad0_sha256.json` |

No winner, no ranking, no new training, no TEST split.
