FROM python:3.11-slim

WORKDIR /workspace

ENV MUJOCO_PATH=/workspace/mujoco

COPY requirements.txt .
RUN apt-get update && apt-get install -y swig build-essential
RUN pip install --no-cache-dir -r requirements.txt
COPY . /workspace

# Install known base requirements
#RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
#RUN pip install stable-baselines3 tensorboard 'mujoco<3' imageio pyyaml tyro swig
#RUN pip install gymnasium[box2d]