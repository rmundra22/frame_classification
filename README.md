# Project

This repository contains a working base project you may use for writing your solution.

## Prerequisities

All packages necessary to build the project should be defined inside poetry files. Docker installs them automatically.

### Docker Container

Building the docker image:
```
cd vv-ai-assignment
docker build . -t vv-ai-assignment

```

Running the docker container:
```
docker run --gpus all --ipc=host -v $(pwd):/workdir/code -v <path-to-data>:/workdir/data -it vv-ai-assignment bash
```

The above command maps the current working directory from the host OS to the _/workdir/code_ directory inside the docker container as well as host OS _path-to-data_ to the _/workdir/data_ directory inside the docker container.

## Building

Execute the following steps inside the docker container:
```
cd /workdir
poetry shell

```

Now you can run python with all necessary dependencies.

## Model training, testing and inference

The following command runs the model training inside container:
```
PYTHONPATH=${PWD} python ./src/train.py
PYTHONPATH=${PWD} python ./src/test.py
PYTHONPATH=${PWD} python ./src/inference.py

```

## Implementation

Files [train.py](src/train.py "train.py"), [test.py](src/test.py "test.py"), [inference.py](src/inference.py "inference.py") contain `#CANDIDATE` instructions. Each is designed for three difficulty levels (one for each level of ). Follow them
