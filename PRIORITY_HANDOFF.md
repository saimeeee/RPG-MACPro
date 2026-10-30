# Give implementation B priority at A's current map boundaries

User instruction on 2026-10-30: let each A seed finish its current map and save
a complete checkpoint, pause A, prioritize B, then continue A when B completes.

- A: `saimeeee/RPG-MACPro`, seeds 0/1/2, currently training `13m`.
- B: `saimeeee/MACPro`, seed 0 running, seeds 1/2 queued.
- Both retain `5m_vs_6m -> 13m -> 4m -> 8m_vs_9m` and budgets
  1M / 0.5M / 0.5M / 0.5M. No algorithm settings change.

`scripts/prioritize_rpg_b.py` is a detached Linux coordinator. It watches each
seed's `checkpoints/task_01/driver.json` and `learner.th`, validates matching
configuration, completed tasks, counters, optimizer, target networks,
continual-learning archives and RNG state, then makes a byte-identical copy.
Only after validation does it stop that seed's verified process group. The
existing B launcher observes the released resources and starts its queued seeds.

The coordinator requires B seeds 0/1/2 to exit successfully, provide valid full
`completion.json` records, and release their process groups before resuming A.
It verifies that A's source files and copied checkpoints have not changed, then
starts A from those checkpoints in fresh result directories. Completed A maps
are not retrained. It monitors the resumed A processes through final completion.
It refuses duplicate installation, existing output directories, PID reuse,
incomplete checkpoints, and mixed-source resumption. Failures are recorded as
`needs_attention`; it never silently starts from zero or retries training.

## Server layout

- Control: `/root/autodl-tmp/rpg_b_priority_20261030`
- Authoritative status: control directory's `manifest.json`
- Coordinator log: control directory's `supervisor.log`
- Preserved checkpoints: control directory's `seed_N/checkpoint`
- Current A output: `/root/autodl-tmp/rpg_smac_buffer5000/seed_N/marines/attempt_000`
- Resumed A output: `/root/autodl-tmp/rpg_smac_buffer5000/seed_N/marines/after_b/attempt_000`
- B status: `/root/autodl-tmp/rpg_alternate_marines/manifest.json`
- Status pointers: `priority_handoff.json` under A's output root and the previous
  `/root/autodl-tmp/rpg_buffer5000_marines_transition` control directory.

The older transition supervisor remains a historical record. It may mark an
intentionally stopped source attempt `failed`; the priority-handoff manifest
records the deliberate pause and subsequent continuation. Existing output files
and earlier checkpoints are preserved. The poll interval is one second while
waiting for A; any incidental next-map work after a boundary and before process
termination is superseded by the preserved boundary, not counted twice.

Run only once, with A's existing Python environment and environment variables:

```bash
python -u prioritize_rpg_b.py \
  --transition /root/autodl-tmp/rpg_buffer5000_marines_transition \
  --b-output /root/autodl-tmp/rpg_alternate_marines \
  --control /root/autodl-tmp/rpg_b_priority_20261030 \
  --target-stage 1
```

The coordinator survives SSH disconnection when started in a detached session.
It does not survive a server shutdown; an existing manifest then requires an
explicit state review before recovery. SMAC simulator RNG is absent from the
existing checkpoint format, so resumed simulator trajectories are not promised
bitwise identical to an uninterrupted run. Model and saved training state are
preserved byte for byte.

Validation: tests cover checkpoint integrity, continuing the next map from a
real miniature learner, invalid checkpoints, PID reuse, independent seed
boundaries, all-three-B completion gating, duplicate-launch prevention, and
Linux termination of a controlled worker plus its simulated environment child.

## Verified installation

At **2026-10-30 16:46:33 Asia/Tokyo**, coordinator PID **45913** was alive with
a 0.26-second heartbeat age and status `waiting_for_a_boundaries`. A seeds
0/1/2 were still training `13m` at 258790 / 266685 / 258497 stage steps; all
three targets are the completed `task_01` checkpoint. B seed 0 remained running
and seeds 1/2 remained pending. No A training process was stopped at installation.
The Linux server passed **all 9 handoff tests**. Local related checks passed
15 tests, with 3 Linux-specific tests skipped on Windows. Installation evidence
is stored in the server control directory's `installation_verified.json`.
