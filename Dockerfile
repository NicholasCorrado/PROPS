FROM python:3.10

WORKDIR /workspace
COPY . /workspace

# Install known base requirements
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install stable-baselines3 tensorboard 'mujoco<3' imageio pyyaml tyro swig
RUN pip install gymnasium[box2d]