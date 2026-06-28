# mrt2-webui

### Realtime Web UI for Magenta RT2

Stream AI-generated music over WebSocket —  
steer the style with **text**, **images**, **video**, or **🖥️ your screen**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.10.1-A259FF?style=flat-square)](https://jax.readthedocs.io)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)

</div>

---

## What is this?

[Magenta RT2](https://github.com/magenta/magenta-realtime) is Google's realtime music generation model. This project wraps it in a browser-based UI so you can:

- 🎵 **Stream live audio** directly in your browser via WebSocket
- 💬 **Guide the style** using natural language prompts
- 🖼️ **Drop in images or video** to set the mood visually
- 🖥️ **Share your screen** and let the music react to what's on it

No command line required once it's running.

---

## Requirements

| Component | Version |
|-----------|---------|
| JAX | 0.10.1 |
| CUDA | 12.8 |
| cuDNN | 9.10.2 |

---

## Installation

### 1. Set up Magenta RT2

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
pip install -e ".[mlx]"
```

```bash
export MAGENTA_HOME=/mnt/Magenta
mrt models init
mrt models download
mrt checkpoints download
```

### 2. Install mrt2-webui

```bash
git clone https://github.com/yhs3048/mrt2-webui.git
cd mrt2-webui
pip install -r requirements.txt
```

---

## Usage

> _(Add your launch command here, e.g. `python app.py` or `uvicorn main:app --reload`)_

Open your browser and navigate to `http://localhost:PORT` — the UI will handle the rest.

---

## How it works

```
Browser  ──WebSocket──▶  mrt2-webui server  ──▶  Magenta RT2 (JAX/GPU)
   ▲                           │
   │   text / image / screen   │  audio stream
   └───────────────────────────┘
```

Style inputs (text, image, video, screen capture) are forwarded to the RT2 model in realtime. Generated audio is streamed back to the browser as it's produced.

---

## Acknowledgements

Built on top of [Magenta RT2](https://github.com/magenta/magenta-realtime) by the Google Magenta team.

---

<div align="center">
  <sub>Made with 🎛️ by <a href="https://github.com/yhs3048">yhs3048</a></sub>
</div>
