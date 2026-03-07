# Description

This agent enables running Prime Intellect [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, including many in Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=by_sections) in NeMo Gym. 

## Install Gym

```
git clone https://github.com/NVIDIA-NeMo/Gym
cd Gym
uv venv; source .venv/bin/activate; uv sync
```

## Test acereason-math example 

First set `env.yaml`, for example for a vLLM served model:
```
policy_base_url: "http://localhost:8000/v1"
policy_api_key: EMPTY
policy_model_name: "Qwen/Qwen3-4B-Instruct-2507"
```

```
# start nemo gym servers
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/acereason-math.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# generate a rollout
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl \
    +limit=1

# view the rollout
tail -n 1 responses_api_agents/verifiers_agent/data/acereason-math-example-rollouts.jsonl | jq | less
```


## Testing new prime environments from environments hub

Some examples: `primeintellect/acereason-math`, `primeintellect/ascii-tree` and `primeintellect/alphabet-sort`.

### Install an environment
```
# deactivate the main nemo gym virtual environment
deactivate

cd responses_api_agents/verifiers_agent

# note that you must have already ran ng_run such as above
# for this .venv to be created in responses_api_agents/verifiers_agent
source .venv/bin/activate

# install the env to create a dataset
uv add tool prime 
prime env install primeintellect/ascii-tree
```

### Create dataset
```
python3 scripts/create_dataset.py --env-id primeintellect/ascii-tree --size 5 --output data/ascii-tree-example.jsonl
```

### Update agent server requirements
```
-e nemo-gym[dev] @ ../../
verifiers==0.1.9.post3
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
ascii-tree
```
### Update agent config
Create `configs/ascii-tree.yaml`, primarily updating env id, and any other env specific args: 
```
verifiers_agent:
  responses_api_agents:
    verifiers_agent:
      entrypoint: app.py
      model_server:
        type: responses_api_models
        name: policy_model
      model_name: ""
      vf_env_id: ascii-tree
      vf_env_args: {}
      group_size: 1
      max_concurrent_generation: -1
      max_concurrent_scoring: -1
      max_tokens: 8192
      temperature: 1.0
      top_p: 1.0

```

```
# return to Gym/ root and activate Gym/ virtual environment
cd ../../
deactivate
source .venv/bin/activate

# start nemo gym servers
ng_run "+config_paths=[responses_api_agents/verifiers_agent/configs/ascii-tree.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# generate a rollout
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example.jsonl \
    +output_jsonl_fpath=responses_api_agents/verifiers_agent/data/ascii-tree-example-rollouts.jsonl \
    +limit=1
```

## Integration notes

The patch to include prompt and generation token ids for preventing retokenization error when training with NeMo RL works specifically with the pinned verifiers version. In newer version of verifiers, this may have change. Thus, we need to make sure to use the pinned version of verifiers and environments that are compatible with this version.

Also, we are using the `.venv` in the `verifiers_agent` folder for installing new environments for generating datasets to avoid dependency conflicts with the `exclude-dependencies` section of Gym `pyproject.toml`. After installing an environment in this `.venv`, make sure to restart NeMo Gym servers with `ng_run` in order to reinstall the pinned version of verifiers, in case changed during environment installation for datataset prep. 

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
