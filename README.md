# mrt2-webui
Realtime web UI for Magenta RT2 — stream music over WebSocket and steer the style with text, images, video, or your screen.

Developed from JAX 0.10.1, CUDA 12.8, cuDNN 9.10.2

git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
pip install -e ".[mlx]"
export MAGENTA_HOME=/mnt/Magenta
mrt models init
mrt models download
mrt checkpoints download

git clone https://github.com/yhs3048/mrt2-webui.git
cd mrt2-webui
pip install -r requirements.txt
