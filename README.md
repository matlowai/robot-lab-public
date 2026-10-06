# Blackforge Robot Lab (public snapshot)

Course log and reference code from a home robot-learning lab: a simulated security-patrol robot moving
through a crowd, with reinforcement learning run the honest way (pre-registered, sealed test seeds), and an
SO-101 arm learning to put objects in a bowl in Isaac Lab 3.0.

- **Course site:** <https://matlowai.github.io/robot-lab-public/> (source: [`docs/index.html`](docs/index.html)).
  It carries Part II (Patrol Lab) and the start of Part I's redo: the SO-101 arm with NVIDIA GR00T N1.7
  ([`#arm`](https://matlowai.github.io/robot-lab-public/#arm)), from 8/72 to 73/144 strict simulated successes in two nights.
- **Talk ("Machines Among Us"):** <https://matlowai.github.io/robot-lab-public/talk/>: the opening reel,
  a 29-slide deck and an interactive field guide (state as of 2026-10-01). Some of its media are FLUX 3
  Action outputs under a non-commercial license; see [`docs/talk/MEDIA-LICENSE.md`](docs/talk/MEDIA-LICENSE.md).
- **Reference code:** [`code/robot-lab/`](code/robot-lab/): the SO-101 Isaac Lab tasks (`robot_lab/tasks/`), the
  scripted expert and recorder, LeRobot conversion, evaluation and the probes used to debug the scene.
  Start with [`PRIMER.md`](code/robot-lab/PRIMER.md). The GR00T N1.7 pipeline is `tools/gr00t_*.py` with the
  overnight drivers `tools/overnight_gr00t.sh` (night 1) and `tools/overnight_gr00t_v2.sh` (night 2: release-tail
  demos, weighted fine-tune, strict eval, report). [`code/robot-lab/rl/`](code/robot-lab/rl/) is residual RL with a
  learned release gate on a frozen GR00T policy (see its README and DESIGN).
  [`code/robot-lab/verifier/`](code/robot-lab/verifier/) builds labelled frames from the recordings, fine-tunes
  Gemma 4 12B with 4-bit QLoRA (Unsloth) as an "is it in the bowl?" checker, and scores it against sim truth.
- **Reference code:** [`code/patrol-lab/`](code/patrol-lab/): the patrol simulation behind Part II
  (events and incident scoring, scenarios, the Spot controller and local planner, the avoidance bake-off
  with PPO / PPO-Lagrangian RL). `uv run pytest` runs the CPU tests; the RL tests need torch.

This is a clean snapshot, not the working repository: no history, logs, run outputs or planning notes.
Taken on 2026-10-04 from the private repos at robot-lab `d25c78f` and patrol-lab `d05e441`; the GR00T tools,
`rl/` and `verifier/` were added on 2026-10-06 from the robot-lab working tree.
The NORI A3 robot integration is left out because its robot description is licensed CC BY-NC-SA.

Paths such as `/mnt/work/...` and `/mnt/weights/...` in the course and code are where things lived on
the lab machine. Change them to match your setup.

## Requirements (as used)

Ubuntu 26.04, Isaac Sim 6.1 / Isaac Lab 3.0, 2x RTX PRO 6000, LeRobot v3 datasets. The older `code/robot-lab`
policy tools were written against FLUX 3 Action SO-101, whose weights are non-commercial and not included. The
`gr00t_*` tools and `rl/` use NVIDIA Isaac-GR00T and GR00T-N1.7-3B (weights not included). Check the license yourself: the model card says
NVIDIA Open Model License, but the LICENSE file shipped with the 3B checkpoint is the non-commercial NVIDIA License.
The code imports Isaac Lab and runs inside its Python environment; it is reference material, not a
packaged library.

## License

Apache-2.0 for the code, course and talk text in this repository ([LICENSE](LICENSE), [NOTICE](NOTICE)),
**except** the media listed in [`docs/talk/MEDIA-LICENSE.md`](docs/talk/MEDIA-LICENSE.md): FLUX 3 Action outputs
(FLUX Kommunity License, non-commercial) and GR00T N1.7 rollouts (treated as non-commercial under the NVIDIA License
shipped with the checkpoint). Third-party software and model weights keep their own
licenses and are not included.
