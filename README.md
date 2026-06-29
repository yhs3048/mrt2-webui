# mrt2-webui

### Realtime Web UI for Magenta RT2

<img width="690" height="500" alt="image" src="https://github.com/user-attachments/assets/ff9d8908-316f-4627-8eb2-96752f432dff" />


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

---

## Installation

### 1. Set up Magenta RT2

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
pip install -e ".[mlx]"
```

```bash
export MAGENTA_HOME=/any_folder
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

### 3. Folder structure

Both repositories must sit side by side in the same parent directory.

```
anywhere/
├── magenta-realtime/     # Magenta RT2 core
└── mrt2-webui/           # This repo
```

> `MAGENTA_HOME` is where model weights are stored and can be set to any path independently.

---
 
## Usage
 
```bash
python webui.py
```
 
Then open your browser and navigate to `http://localhost:8000`.
 
### Options
 
| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8000` | Port to serve the web UI on |
| `--clip-device` | `cuda` | Device for the CLIP encoder — `cpu`, `cuda`, or a specific GPU like `cuda:1` |

> **Note:** Ensure port `8000` (or your chosen `--port`) is open and accessible.
 
---

## Acknowledgements

Built on top of [Magenta RT2](https://github.com/magenta/magenta-realtime) by the Google Magenta team.

Developed by [yhs3048](https://github.com/yhs3048) at Music and Audio Research Group (MARG), Seoul National University.

---
