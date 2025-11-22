# DROID Sim Evaluation

**Note:** The current simulator works best for policies trained with *joint position* action space (and *not* joint velocity control). 


## Installation

Clone the repo
```bash
git clone --recurse-submodules git@github.com:jca0/pi-sim-evals.git
cd pi-sim-evals
```

Install uv (see: https://github.com/astral-sh/uv#installation)

For example (Linux/macOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate virtual environment
```bash
uv sync
source .venv/bin/activate
```

## Quick Start

First, make sure you download the simulation assets and unpack them into the root directory of this package.
Using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), this can be done with:
```bash
curl -O https://pi-sim-assets.s3.us-east-1.amazonaws.com/assets.zip 
unzip assets.zip
```

Then, in a separate terminal, launch the policy server on `localhost:8000`. 

For example, to launch a pi0-FAST-DROID policy (with joint position control),
checkout [openpi](https://github.com/Physical-Intelligence/openpi/tree/karl/droid_policies) to the `karl/droid_policies` branch and run the command below in a separate terminal
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos
```

To launch a pi0.5-DROID policy, checkout [openpi](https://github.com/Physical-Intelligence/openpi/tree/main) to the `main` branch and run the command below in a separate terminal
``` bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```
pi0.5-DROID outputs joint velocities, but the simulation script converts to joint positions.

**Note**: We set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to avoid JAX hogging all the GPU memory (since Isaac Sim needs to use the same GPU).

Finally, run the evaluation script:
```bash
python run_eval.py --episodes [INT] --scene [INT] --headless --policy [pi0.5, pi0]
```

## Minimal Example

```python
env_cfg.set_scene(scene) # pass scene integer
env = gym.make("DROID", cfg=env_cfg)

obs, _ = env.reset()
obs, _ = env.reset() # need second render cycle to get correctly loaded materials
client = # Your policy of choice

max_steps = env.env.max_episode_length
for _ in tqdm(range(max_steps), desc=f"Episode"):
    action = client.infer(obs, INSTRUCTION) # calling inference on your policy
    action = torch.tensor(ret["action"])[None]
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
env.close()
```
