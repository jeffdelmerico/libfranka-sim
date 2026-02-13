Install libfranka and dependencies on Ubuntu 24.04:
```
sudo apt install libpoco-dev libfmt-dev libeigen3-dev
wget https://github.com/frankarobotics/libfranka/releases/download/0.20.4/libfranka_0.20.4_noble_amd64.deb
wget https://github.com/frankarobotics/libfranka/releases/download/0.20.4/libfranka_0.20.4_noble_amd64.deb.sha256
sha256sum -c libfranka_0.20.4_noble_amd64.deb.sha256
sudo dpkg -i libfranka_0.20.4_noble_amd64.deb
```

Install libfranka-sim and deps:
```
git clone https://github.com/jeffdelmerico/libfranka-sim.git
cd libfranka-sim
git checkout -b mujoco
uv venv --python 3.11
source .venv/bin/activate
uv pip install mujoco numpy numba robot_descriptions pylibfranka
```

Ubuntu 20.04
RT kernel additional steps:
```
gpg2 --locate-keys torvalds@kernel.org gregkh@kernel.org
gpg2 --tofu-policy good 38DBBDC86092693E
gpg2 --trust-model tofu --verify linux-*.tar.sign
```
After installing kernel deb pkg, needed to go into BIOS and disable Secure Boot.

Validate real-time metrics:
```
sudo apt install rt-tests
sudo cyclictest --mlockall --smp --priority=80 --interval=200 --distance=0
```

```
sudo apt install libfmt-dev
wget https://github.com/frankarobotics/libfranka/releases/download/0.20.4/libfranka_0.20.4_focal_amd64.deb
sudo dpkg -i libfranka_0.20.4_focal_amd64.deb 
```

```
git clone https://github.com/jeffdelmerico/libfranka-sim.git
cd libfranka_sim
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --python 3.9
uv venv
source .venv/bin/activate
uv pip install pylibfranka mujoco==3.3.7
uv pip install -e .
```
