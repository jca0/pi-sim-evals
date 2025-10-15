# openpi notes
- sometimes policy server won't launch. solution is to run in virtual environment *without* uv
**command run:**  
```
cd ~/openpi
. .venv/bin/activate
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```

# pi-sim-eval notes
- you can't run it on a mac, it needs CUDA 12.xx
- added IsaacSim submodule to repo
