# sim-evals

## Installation

Clone repo
```bash
git clone --recurse-submodules git@github.com:arhanjain/sim-evals.git
cd sim-evals
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

Install extra dependencies
```bash
./submodules/IsaacLab/isaaclab.sh -i
```

## Quick Start
With environment active run:
```
python main.py  
```

