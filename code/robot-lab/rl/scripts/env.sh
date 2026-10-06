# Shared environment for robot-lab RL (source me). Paths match tools/overnight_gr00t.sh.
NA=/mnt/weights/ai/nvidia-action
GR=$NA/Isaac-GR00T
GRPY=$GR/.venv/bin/python
ISAAC=/mnt/weights/ai/isaac/IsaacLab
RL=/mnt/work/AI/robot-lab/rl
DATA=/mnt/weights/ai/robot-lab-data/gr00t-rl
FF7=$NA/ffmpeg7/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1/lib
UNITS=/mnt/weights/ai/robot-lab-data/overnight/full-20260923-2207/lerobot_ds/units.json
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export HF_HOME=$NA/hf HF_HUB_CACHE=$NA/hf/hub HF_XET_CACHE=$NA/hf/xet TMPDIR=$NA/tmp UV_CACHE_DIR=$NA/uv-cache
GRENV=(env LD_LIBRARY_PATH="$FF7" PATH="$GR/.venv/bin:$PATH")   # torchcodec 0.8 needs FFmpeg <= 7
ISAACPY=$ISAAC/.venv/bin/python
