# HassoscopeProPy

## Preliminaries

Install [uv](https://docs.astral.sh/uv/). 

To do so on Linux or macOS run following in the terminal

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows run following in the Powershell window:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Normally this will apply change to your PATH env variable but sometimes that fails and you may need to 
update it yourself. Application `uv` will be found at either `~/.local/bin/uv` or `~/.cargo/bin/uv` depending on
installation method.

## Setup repo

```shell
git clone git@github.com:priimak/HaasoscopeProPy.git
cd HaasoscopeProPy/
uv venv -p 3.13
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Note, that this code uses python 3.13, however you don't have to download it if you do not have installed 
yet, `uv` will download and install somewhere under `~/.local/` if it is not already installed.

## Running example application

Assuming that you went through steps listed above you can run [example.py](https://github.com/priimak/HaasoscopeProPy/blob/master/src/example.py) 
from the terminal like so:

```shell
uv run --directory src example.py
```

