# openpi notes
- sometimes policy server won't launch. solution is to run in virtual environment *without* uv
**command run:**  
to run pi0_fast_droid in karl/droid_policies branch
``` bash
cd ~/openpi
. .venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```

to run pi0.5_droid in main branch
``` bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

# pi-sim-eval notes
- you can't run it on a mac, it needs CUDA 12.xx
- added IsaacSim submodule to repo
``` bash
. .venv/bin/activate
python3 run_eval.py --episodes 3 --scene 1 --headless --policy pi0.5
```

# ec2 instance
- g6.2xlarge (32GiB)
- 200GiB SSD
- sudo apt install ffmpeg, unzip

# if JAX error, clear cache:
```
rm -rf ~/.cache/jax
rm -rf ~/.cache/xla
```