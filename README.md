# PROPS
Proximal Robust Policy On-Policy Sampling

## Local Installation

```commandline
cd PROPS
pip install -e .
python ppo_discrete.py
```

## Building Docker image for CHTC
1. Install docker and create an account on Docker Hub.
2. Edit `docker_build.sh` to use your Docker Hub username and an appropriate tag for your experiments.
3. Run `docker_build.sh`
