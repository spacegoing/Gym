# Atropos Integration for Nemo Gym

Integration with [Atropos](https://github.com/NousResearch/atropos) using the **Trajectory API** for full multi-turn trajectory collection.

## Architecture

**Correct approach using Trajectory API:**

```
Gym launches resources_servers/atropos/app.py (standard resources server)
   ↓ reads policy_base_url from env.yaml
   ↓ launches Trajectory API (localhost:8000)
   ↓ launches Atropos env server pointing to Gym's vLLM
   ↓ background task pulls batches from API

Atropos Env:
   ↓ calls Gym's vLLM for inference
   ↓ runs collect_trajectories() with multi-turn logic
   ↓ scores & tokenizes
   ↓ POSTs to Trajectory API

app.py seed_session():
   ↓ pulls next trajectory from queue
   ↓ returns it (already scored!)
```

**Key points:**
- ✅ Full multi-turn trajectory support (handled by Atropos)
- ✅ Inference uses Gym's vLLM from **env.yaml**
- ✅ All scoring/tokenization done by Atropos environments
- ✅ Trajectory API queues and batches data
- ✅ Standard resources server pattern

## Quickstart (GSM8k)

**1. Configure env.yaml:**
```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
```

**2. Start Gym's vLLM:**
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --dtype auto \
    --port 10240 \
    --host 0.0.0.0
```

**3. Run Gym:**
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/atropos/configs/gsm8k.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts \
    +agent_name=gsm8k_atropos_agent \
    +input_jsonl_fpath=resources_servers/atropos/data/gsm8k_sample.jsonl \
    +output_jsonl_fpath=results/gsm8k_rollouts.jsonl \
    +limit=5
```

The resources server automatically:
- Launches Trajectory API
- Launches GSM8k environment server (using your vLLM)
- Pulls scored trajectories from API
- Returns them via seed_session()

## Configuration

### env.yaml (Required)
```yaml
# Your vLLM model server
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
```

The resources server reads this config and passes it to Atropos environments.

### Resources Server Config

Example: `configs/gsm8k.yaml`

```yaml
gsm8k_atropos_resources_server:
  resources_servers:
    atropos:
      entrypoint: app.py
      config:
        # Atropos repo path
        atropos_path: /raid/home/cmunley/Gym/atropos

        # Which environment to use
        environment_module: environments.gsm8k_server
        environment_class: GSM8kEnv

        # Environment settings
        group_size: 8  # Number of trajectories per group

        # Optional: custom environment args
        env_args: {}
```

## Supported Environments

Works with **any** Atropos environment:

### GSM8k (Math Reasoning)
```yaml
environment_module: environments.gsm8k_server
environment_class: GSM8kEnv
```

### MCP (Tool Calling)
```yaml
environment_module: environments.mcp_env
environment_class: McpEnv
```

### Multi-Turn Tool Calling
```yaml
environment_module: environments.tool_use_multiturn_server
environment_class: MultiTurnToolCallingEnv
```

### Letter Counting
```yaml
environment_module: environments.letter_counting_environment
environment_class: LetterCountingEnv
```

### KernelBench (CUDA)
```yaml
environment_module: environments.kernelbench_env.kernelbench_env
environment_class: KernelBenchEnv
```

## How It Works

### Resources Server Lifecycle

**On Startup:**
1. Reads `policy_base_url` from env.yaml
2. Launches Trajectory API on port 8000
3. Launches Atropos environment server configured to use Gym's vLLM
4. Waits for environment to start generating data
5. Starts background task to pull batches from API

**On seed_session():**
1. Waits if trajectory queue is empty
2. Returns next pre-scored trajectory from queue
3. Trajectory already has tokens, scores, masks, etc.

**On verify():**
1. Returns the pre-computed reward from the trajectory
2. Optionally includes full trajectory data

### Multi-Turn Support

Unlike simple approaches, this **fully supports multi-turn trajectories**:

**Example: Tool Calling**
```
1. Env generates: "I need to call the weather tool"
2. Env executes: weather_tool("San Francisco") → "72°F, sunny"
3. Env generates: "The weather in SF is 72°F and sunny"
4. Env scores: Did it use the right tool? Correct answer?
5. Env pushes: Complete scored trajectory to API
```

All multi-turn logic handled by Atropos environment's `collect_trajectories()` method.

## Trajectory Format

Trajectories pulled from the API contain:

```python
{
    "tokens": [[int, ...], ...],           # Tokenized sequences
    "masks": [[int, ...], ...],            # Attention masks (1=train, 0=ignore)
    "scores": [float, ...],                # Rewards per trajectory
    "advantages": [[float, ...], ...],     # Optional advantage values
    "ref_logprobs": [[float, ...], ...],   # Optional reference logprobs
    "generation_params": {...},            # Generation config used
    "env_id": int,                         # Which environment generated this
}
```

This format is **ready for RL training** - no additional processing needed.

## Configuration Options

### AtroposServerConfig
- `atropos_path`: Path to Atropos repo (required)
- `environment_module`: Python module path (e.g., "environments.gsm8k_server")
- `environment_class`: Class name (e.g., "GSM8kEnv")
- `group_size`: Number of trajectories per group (default: 8)
- `env_args`: Dict of additional environment-specific args
- `trajectory_api_port`: Port for Trajectory API (default: 8000)
- `api_startup_wait`: Seconds to wait for API startup (default: 10)
- `env_startup_wait`: Seconds to wait for env to generate data (default: 60)

## Files

- `app.py` - Resources server with Trajectory API integration
- `configs/gsm8k.yaml` - GSM8k configuration example
- `configs/tool_calling.yaml` - Multi-turn tool calling example
- `scripts/` - Data preparation utilities

## Why Trajectory API?

The Trajectory API is the **correct** way to use Atropos because:

1. **Multi-turn support**: Environments handle complex multi-turn logic
2. **Async generation**: Multiple envs can generate in parallel
3. **Smart batching**: API handles heterogeneous batch packing
4. **Environment isolation**: Each env is independent, can restart
5. **Flexible allocation**: Can weight different environments differently
6. **No code duplication**: All environment logic stays in Atropos

This is how Atropos is designed to work for RL training!

## Troubleshooting

**Error: "Must set policy_base_url"**
- Add `policy_base_url` to your `env.yaml` file
- It should point to your vLLM server (e.g., `http://localhost:10240/v1`)

**Waiting forever for trajectories**
- Check that vLLM is running on the port specified in env.yaml
- Check logs: Atropos environment should be generating data
- Increase `env_startup_wait` if environment needs more time

**Trajectory API failed to start**
- Ensure `run-api` command is available (install Atropos)
- Check that port 8000 is not already in use
