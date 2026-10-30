# Replay-5000 run: change the untrained maps at the first checkpoint

User request: make the already-running RPG-MACPro replay-5000 experiment use **5m_vs_6m → 13m → 4m → 8m_vs_9m**, matching the separately deployed alternate implementation. Budgets remain **1M / 0.5M / 0.5M / 0.5M**, seeds **0/1/2**, and all algorithm hyperparameters remain unchanged.

At inspection on 2026-10-30 11:43:59 Asia/Tokyo, the three seeds were still training the common first map at approximately 295,691 / 289,858 / 287,667 steps. There were no completed-task checkpoints yet. Stopping them immediately would lose the unfinished first task, so the transition waits for each seed's complete `task_00` checkpoint independently.

`scripts/migrate_rpg_curriculum.py` accepts only the original four-map replay-5000 configuration at the completed first-task boundary. It copies the checkpoint and changes exactly three metadata entries: `config.tasks`, online-agent `_extra_state.task_keys`, and target-agent `_extra_state.task_keys`. Future task embeddings remain in the same t1/t2 slots. It verifies equality of every remaining tensor, optimizer/target state, RNG state, training counter, regularization archive, and first-task result. Source files are preserved. A checkpoint that has learned an additional task is rejected. SMAC simulator RNG is not part of the existing checkpoint format; resumed evaluations are not promised bitwise identical to uninterrupted simulator execution.

`scripts/transition_rpg_marines.py` watches the live experiment. After a first-task checkpoint is complete and the copied conversion passes validation, it verifies the source process PID, start time, command and process group, stops that seed's old continuation, and launches the converted checkpoint on the new maps. It never signals unrelated jobs. The other implementation continues running. A failure is recorded rather than silently restarting from zero.

The continuation receives a fresh output under the same seed prefix, preserving the other launcher's resource accounting:

- Original source attempt: `/root/autodl-tmp/rpg_smac_buffer5000/seed_N/attempts/attempt_000`
- Continued training: `/root/autodl-tmp/rpg_smac_buffer5000/seed_N/marines/attempt_000`
- Authoritative transition status: `/root/autodl-tmp/rpg_buffer5000_marines_transition/manifest.json`
- Converted checkpoints and provenance: `/root/autodl-tmp/rpg_buffer5000_marines_transition/seed_N/checkpoint`
- Pointer from the original experiment root: `marines_transition.json`

The original launcher may label intentionally superseded attempts as failed when they are stopped at the boundary. Use the transition manifest for the continuing experiment; retain the original manifest and attempt logs as historical evidence. The completed replay-256 results are not relabeled or changed, and their different intermediate maps cannot be directly compared with this new curriculum.

The standalone task config for new runs is `src/config/tasks/rpg_marines_macpro.yaml`; the historical paper-sequence config remains available.

## Verified deployment

The server passed all **9 migration/supervisor tests**, including preservation of learned state, training on the renamed second task, end-to-end supervision with controlled process doubles, and refusal to signal a reused PID. The original implementation also completed a real four-map CUDA short run: **2506 steps, 88 episodes, 80 optimizer updates**, with finite losses and all four task checkpoints, using seed 701 and a separate test output. These are validation runs, not full-experiment performance results.

The detached transition supervisor is **PID 35752**. At **2026-10-30 11:55:41 Asia/Tokyo**, its persisted state was `waiting_for_first_task` for all three seeds. Source training was still running at **341901 / 334102 / 331931** steps, respectively. The alternate implementation's seed-0 process also remained running. Actual continuation has not happened at this snapshot; it is automatically triggered by each source's completed first-task checkpoint. The status and validation snapshot is archived locally under `results/server-marines-transition/`.
