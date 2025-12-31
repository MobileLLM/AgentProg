# Evaluation on AndroidWorld and AW-Extend

## Environment Setup

Please refer to the official [guide](./android_world/README.md) to setup AndroidWorld Benchmark.

## Run Evaluation

You can configure your adb console port and grpc port of emulator [here](./android_world/node_config.yaml), which supports multi-process evaluation of AndroidWorld.

For Evaluation of original AndroidWorld Benchmark, run:

```python
EXP_NAME=<your experiment name> PROCESS_ID=0 LONG_TASKS=0 python run_agentprog.py 
```

For Evaluation of AW-Extend Benchmark, run:

```python
EXP_NAME=<your experiment name> PROCESS_ID=0 LONG_TASKS=1 python run_agentprog.py 
```