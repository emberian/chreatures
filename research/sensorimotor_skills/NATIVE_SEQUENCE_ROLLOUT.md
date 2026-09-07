# CNS-only native rollout boundary

Status: **host primitives implemented; PPO collection intentionally inactive**.

The current resident accepts only an authenticated CNS latent `float32[B,512]`,
the previous delivered command `float32[B,12]`, physical ticks `uint64[B]`, and
reset flags. Raw optics, physical observations, physiology, reward, and the old
384-neuron readout never enter the recurrent controller. The old 668-coordinate
rollout proposal has been removed rather than retained as a compatibility path.

`native_sequence_rollout.py` contains the reusable causal boundary for a later
rollout contract. It calls the current native resident, gives the proposed
command to the existing physical barrier, and calls the explicit native
acknowledgement with the command the world actually delivered. A row becomes
available to a caller only after that acknowledgement succeeds. Any exception
after entry to the physical barrier is reported as an uncertain mutation and is
never retried. Read-only tail evaluation uses the native private clone through
`preview_sequence_control`.

The native acknowledgement leaves `cns_outcome_pending=true`: the next
authenticated CNS step resolves the causal effect of the delivered command.
This is coherent continuation state, while `command_pending` must be false after
the receipt. Native snapshots retain the recurrent state, goal memory, acquired
motor suffixes, policy weights and RNGs. Only research-training cohorts may swap
the immutable sequence-control head artifact at an acknowledged boundary.

The first CNS adapter wave uses supervised and self-supervised sequential
training. There is no approved CNS-derived policy teaching signal or PPO packet
schema yet, so `ACTIVE_ROLLOUT_CONTRACT` remains `None` and command-line
invocation fails before allocating a world or neural service. A later policy
wave must define the teaching signal, packet fields, coherent whole-system
checkpoint, and manifest identity together before this gate changes.

The resident artifact and CNS service explicitly report
`training_status=initialized-untrained` or `trained`. A fresh executable random
controller is evidence that the full CNS-to-world path runs; it is not evidence
of learned sensorimotor competence.
