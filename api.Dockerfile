FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git curl build-essential \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && ln -s /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY . .

# Optional: if you rely on IsaacLab source/submodule paths at runtime
# ENV PYTHONPATH="/app/submodules/IsaacLab/source:${PYTHONPATH}"

# GPU-related envs commonly used with IsaacSim/IsaacLab containers
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

EXPOSE 9000

CMD ["uvicorn", "sim_server:app", "--host", "0.0.0.0", "--port", "9000"]