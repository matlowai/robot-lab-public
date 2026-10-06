# Media license note for the talk

The talk pages (`index.html`, `deck.html`, `explorer.html`, `reel/reel.html`, `race.js`, `reelplayer.js`)
are Apache-2.0 like the rest of this repository, **except the media files listed below**.

These files are outputs of, or contain outputs of, **FLUX 3 Action** by Black Forest Labs, which is
licensed under the **FLUX Kommunity License (non-commercial)**. They are shared here for non-commercial,
educational use only. They are **not** covered by this repository's Apache-2.0 license, and the model
weights are not included.

- `media/arm_fake_success.mp4`, `media/arm_fake_success_still.jpg` (FLUX SO-101 policy rollout)
- `media/arm_first_success.mp4`, `media/arm_first_success_still.jpg` (FLUX SO-101 policy rollout)
- `media/droid_imagined.mp4`, `media/droid_imagined_still.jpg` (FLUX video prediction)
- `media/t2v_ocean.mp4`, `media/t2v_ocean_still.jpg` (FLUX text-to-video)
- `reel/droid_smooth.mp4` (FLUX video prediction)
- `media/opening_reel.mp4`, `media/opening_reel_silent.mp4`, `media/opening_reel_still.jpg` (contain the clips above)

## NVIDIA GR00T N1.7 rollouts (added 2026-10-06)

The arm clips added on 2026-10-06 show a GR00T N1.7 policy (fine-tuned from NVIDIA's GR00T-N1.7-3B) acting in
simulation. The model card names the NVIDIA Open Model License, but the LICENSE file shipped with the checkpoint is
the **NVIDIA License**, whose section 3.3 limits "the Work and any derivative works thereof" to non-commercial
research or evaluation use. Until that is resolved, we treat these rollout media as non-commercial too: they are
**not** covered by this repository's Apache-2.0 license, and no model weights are included.

- `media/arm_gr00t_never_lets_go.mp4`, `media/arm_gr00t_never_lets_go_still.jpg`
- `media/arm_gr00t_let_go_mug.mp4`, `media/arm_gr00t_let_go_mug_still.jpg`
- `media/arm_gr00t_let_go_cracker_box.mp4`, `media/arm_gr00t_let_go_cracker_box_still.jpg`
- `media/arm_gr00t_rl_base_cracker_box.mp4`, `media/arm_gr00t_rl_base_cracker_box_still.jpg`
- `media/arm_gr00t_rl_gate_cracker_box.mp4`, `media/arm_gr00t_rl_gate_cracker_box_still.jpg`
- `media/arm_gr00t_check_held.jpg`
- `media/opening_reel.mp4`, `media/opening_reel_silent.mp4` (contain the clips above)
- In the course: `../media/part1/seed1004_night1_vs_night2.mp4`, `../media/part1/night2_cracker_box_ep05.mp4`,
  their `_still.jpg` posters and `../media/part1/night2_cracker_box_ep05_strip.jpg`

`media/arm_gr00t_check_settled.jpg` is a scripted-expert demo frame, not model output.

The reel and the deck were updated on 2026-10-06 with the GR00T nights; the field guide is being updated,
and newer results are in the course.
