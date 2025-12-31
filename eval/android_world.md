# Evaluation on AndroidWorld and AW-Extend

## 1. Benchmark Overview

### AndroidWorld
[**AndroidWorld**](https://github.com/google-research/android_world) is a widely used benchmark for evaluating Android GUI agents across diverse real-world mobile applications. It primarily consists of atomic, short-horizon tasks (e.g., "Open Settings," "Add a calendar event").

### AW-Extend
**AW-Extend** is a specialized benchmark extension proposed in the paper ["AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management"](https://arxiv.org/pdf/2512.10371). It is designed to address the limitations of existing mobile-use benchmarks by focusing on Long-Horizon Tasks.

AW-Extend includes 19 complex tasks categorised into two types:
*   **Compositional Tasks**: Workflows that combine multiple atomic operations with strong dependencies (e.g., *"Find Alice's new address in Emails, save it to Contacts, and then send her a confirmation SMS"*).
*   **Iterative Tasks**: High-repetition batch operations (e.g., adding 10 distinct contacts in a row) designed to test an agent's robustness and ability to filter irrelevant historical context.


## 2. Prerequisites

Before running the evaluation, please ensure the AndroidWorld environment is correctly installed. Consult the official [AndroidWorld Setup Guide](./android_world/README.md) for detailed installation instructions.

## 3. Configuration

You can configure the emulator settings (including the ADB console port and gRPC port) to enable multi-process evaluation.

*   **Config File**: [`./android_world/node_config.yaml`](./android_world/node_config.yaml)

Edit this file to match your specific port configurations and parallel processing needs.

## 4. Usage

Run the evaluation script `run_agentprog.py` using the commands below. The benchmark type is controlled by the `LONG_TASKS` environment variable.

### Standard AndroidWorld Benchmark
To evaluate on the original AndroidWorld dataset (short-horizon tasks):

```bash
EXP_NAME=<your_experiment_name> PROCESS_ID=0 LONG_TASKS=0 python run_agentprog.py
```

### AW-Extend Benchmark
To evaluate on the AW-Extend dataset (long-horizon tasks):

```bash
EXP_NAME=<your_experiment_name> PROCESS_ID=0 LONG_TASKS=1 python run_agentprog.py
```

**Arguments:**
*   `EXP_NAME`: A unique identifier for your experiment.
*   `PROCESS_ID`: The ID of the current process (set to `0` for single-process runs). For multi-process evaluation, specify a unique ID to connect to the corresponding emulator defined in the configuration file.
*   `LONG_TASKS`: Set to `0` for standard tasks, `1` for extended long-horizon tasks.
