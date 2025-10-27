# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Atropos Resources Server for Nemo Gym.

This server integrates Atropos environments using the Trajectory API:
1. On startup: launches Trajectory API + Atropos env servers
2. Atropos envs use endpoint in env.yaml for inference
3. Envs run collect_trajectories()
4. This server pulls batches from Trajectory API
5. Maps requests to pre-scored trajectories

Architecture:
    Gym launches this resources server
      - starts Trajectory API (localhost:8000)
      - starts Atropos env servers pointing to endpoint in env.yaml
      - envs push scored trajectories to API
    Gym calls seed_session(task_idx)
      - pulls batch from API
      - returns trajectory as if generated on-demand
"""
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Any
from collections import deque
from contextlib import asynccontextmanager
from socket import socket

import aiohttp
from fastapi import FastAPI, Request
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    with socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.integrations.atropos import (
    AtroposResourcesServerConfig,
    AtroposSeedSessionRequest,
    AtroposSeedSessionResponse,
    AtroposStepRequest,
    AtroposStepResponse,
    AtroposAgentVerifyRequest,
    AtroposAgentVerifyResponse,
    AtroposCloseRequest,
    AtroposCloseResponse,
)
from nemo_gym.global_config import (
    POLICY_BASE_URL_KEY_NAME,
    POLICY_API_KEY_KEY_NAME,
    POLICY_MODEL_NAME_KEY_NAME,
    get_global_config_dict,
)


class AtroposServerConfig(AtroposResourcesServerConfig):
    atropos_path: str
    environment_module: str  # e.g. "environments.gsm8k_server"
    environment_class: str   # e.g. "GSM8kEnv"

    trajectory_api_port: int = 8000
    trajectory_api_url: str = "http://localhost:8000"

    group_size: int = 8
    env_args: Dict[str, Any] = Field(default_factory=dict)

    api_startup_wait: int = 10
    env_startup_wait: int = 60


class AtroposResourcesServer(SimpleResourcesServer):
    config: AtroposServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._api_process: Optional[subprocess.Popen] = None
        self._env_process: Optional[subprocess.Popen] = None

        self._trajectory_queue: deque = deque()
        self._batch_task: Optional[asyncio.Task] = None

        self._sessions: Dict[str, Dict[str, Any]] = {}

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._startup()
            yield
            await self._shutdown()

        app = FastAPI(lifespan=lifespan)
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        app.post("/step")(self.step)
        app.post("/verify")(self.verify)
        app.post("/close")(self.close)

        return app

    async def _startup(self):
        """Launch Trajectory API and Atropos environment servers."""
        print("=== Starting Atropos Servers ===")

        global_config = get_global_config_dict()
        policy_base_url = global_config.get(POLICY_BASE_URL_KEY_NAME, "")
        policy_api_key = global_config.get(POLICY_API_KEY_KEY_NAME, "EMPTY")
        policy_model_name = global_config.get(POLICY_MODEL_NAME_KEY_NAME, "policy_model")

        if not policy_base_url:
            raise ValueError(
                f"Must set {POLICY_BASE_URL_KEY_NAME} in env.yaml or config. "
                "This should point to your vLLM server or openai compatible API (e.g., http://localhost:10240/v1)"
            )

        print(f"Using model server: {policy_base_url}")

        api_port = find_free_port()
        self.config.trajectory_api_port = api_port
        self.config.trajectory_api_url = f"http://localhost:{api_port}"
        print(f"Using Trajectory API port: {api_port}")

        print("[1/3] Launching Trajectory API...")
        try:
            await self._launch_trajectory_api()
            await self._wait_for_api_health()
            await self._register_trainer()
        except Exception as e:
            print(f"ERROR launching Trajectory API: {e}")
            raise

        print("[2/3] Launching Atropos environment server...")
        try:
            await self._launch_env_server(policy_base_url, policy_api_key, policy_model_name)
        except Exception as e:
            print(f"ERROR launching environment server: {e}")
            raise

        print("[3/3] Starting batch collection...")
        self._batch_task = asyncio.create_task(self._batch_collection_loop())

        print(f"Waiting {self.config.env_startup_wait}s for environment to generate data...")
        await asyncio.sleep(self.config.env_startup_wait)

        print("=== Atropos Servers Ready ===")

    async def _shutdown(self):
        print("=== Shutting down Atropos Servers ===")

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        if self._env_process:
            self._env_process.terminate()
            try:
                self._env_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._env_process.kill()

        if self._api_process:
            self._api_process.terminate()
            try:
                self._api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._api_process.kill()

    async def _launch_trajectory_api(self):
        resources_dir = Path(__file__).parent
        run_api_path = resources_dir / ".venv" / "bin" / "run-api"

        if not run_api_path.exists():
            raise FileNotFoundError(
                f"run-api not found at {run_api_path}. "
                "Make sure atropos is installed in the venv (check requirements.txt)."
            )

        cmd = [
            str(run_api_path),
            "--host", "0.0.0.0",
            "--port", str(self.config.trajectory_api_port),
        ]

        print(f"  Command: {' '.join(cmd)}")

        self._api_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        await asyncio.sleep(2)

        if self._api_process.poll() is not None:
            output = self._api_process.stdout.read().decode() if self._api_process.stdout else ""
            print(f"  ERROR: Trajectory API process exited immediately!")
            print(f"  Output: {output}")
            raise RuntimeError(f"Trajectory API failed to start: {output}")

    async def _wait_for_api_health(self):
        print(f"  Waiting for API at {self.config.trajectory_api_url}")

        async with aiohttp.ClientSession() as session:
            for _ in range(self.config.api_startup_wait):
                try:
                    async with session.get(
                        f"{self.config.trajectory_api_url}/status",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            logger.info("  API ready!")
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(1)

        raise TimeoutError("Trajectory API failed to start")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    async def _register_trainer(self):
        """Register with Trajectory API"""
        async with aiohttp.ClientSession() as session:
            data = {
                "wandb_group": "nemo_gym",
                "wandb_project": "nemo_gym",
                "batch_size": 32,
                "max_token_len": 4096,
                "starting_step": 0,
                "checkpoint_dir": "checkpoints",
                "save_checkpoint_interval": 1000,
                "num_steps": 10000,
            }

            async with session.post(
                f"{self.config.trajectory_api_url}/register",
                json=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                print("  Registered with API")

    async def _launch_env_server(self, vllm_url: str, api_key: str, model_name: str):
        """Launch Atropos environment server"""
        atropos_path = Path(self.config.atropos_path)
        script_path = atropos_path / f"{self.config.environment_module.replace('.', '/')}.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Environment script not found: {script_path}")

        import tempfile
        import yaml

        config_data = {
            "env": {
                "group_size": self.config.group_size,
                "rollout_server_url": self.config.trajectory_api_url,
                **self.config.env_args,
            },
            "openai": {
                "base_url": vllm_url,
                "api_key": api_key,
                "model_name": model_name,
                "timeout": 300,
                "num_max_requests_at_once": 8,
            },
            "slurm": False,
        }

        config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_data, config_file)
        config_file.close()

        cmd = [
            "python", str(script_path), "serve",
            "--config", config_file.name,
        ]

        print(f"  Environment: {self.config.environment_class}")
        print(f"  Command: {' '.join(cmd)}")

        # Write output to log files for debugging
        log_dir = Path("/tmp/atropos_env_logs")
        log_dir.mkdir(exist_ok=True)
        stdout_log = log_dir / f"env_{self.config.environment_class}.log"

        with open(stdout_log, "w") as log_file:
            self._env_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        print(f"  Environment logs: {stdout_log}")

        await asyncio.sleep(5)

        if self._env_process.poll() is not None:
            with open(stdout_log, "r") as f:
                output = f.read()
            print(f"  ERROR: Environment server exited immediately!")
            print(f"  Last 50 lines of log:")
            print('\n'.join(output.split('\n')[-50:]))
            raise RuntimeError(f"Environment server failed to start - check {stdout_log}")

    async def _batch_collection_loop(self):
        while True:
            try:
                batch = await self._get_batch_from_api()

                if batch is not None and batch.get("batch"):
                    for group in batch["batch"]:
                        self._trajectory_queue.append(group)

                    logger.debug(f"Pulled batch, queue size: {len(self._trajectory_queue)}")
                else:
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch collection: {e}")
                await asyncio.sleep(5.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    async def _get_batch_from_api(self) -> Optional[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.config.trajectory_api_url}/batch",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def seed_session(
        self,
        request: Request,
        body: AtroposSeedSessionRequest,
    ) -> AtroposSeedSessionResponse:
        """
        Return a scored trajectory from the queue

        Note: task_idx is ignored - we just return the next available trajectory.
        The Atropos environments are generating trajectories continuously.
        """
        timeout = 120
        start_time = time.time()

        while not self._trajectory_queue:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"No trajectories available after {timeout}s. "
                    "Check that Atropos environment is running and generating data."
                )
            await asyncio.sleep(0.5)

        trajectory = self._trajectory_queue.popleft()

        import uuid
        env_id = str(uuid.uuid4())

        scores = trajectory.get("scores", [0.0])
        avg_reward = sum(scores) / len(scores) if scores else 0.0

        self._sessions[env_id] = {
            "trajectory": trajectory,
            "avg_reward": avg_reward,
        }

        return AtroposSeedSessionResponse(
            env_id=env_id,
            obs=[],
            system_prompt=None,
            metadata={
                "trajectory_ready": True,
                "trajectory_data": trajectory,
                "avg_reward": avg_reward,
            },
        )

    async def step(
        self,
        request: Request,
        body: AtroposStepRequest,
    ) -> AtroposStepResponse:
        """Step is a no-op - trajectory already done"""
        return AtroposStepResponse(
            obs=[],
            reward=0.0,
            done=True,
            info={},
        )

    async def verify(
        self,
        request: Request,
        body: AtroposAgentVerifyRequest,
    ) -> AtroposAgentVerifyResponse:
        env_id = body.response.env_id
        session = self._sessions.get(env_id, {})

        return AtroposAgentVerifyResponse(
            **body.model_dump(),
            reward=session.get("avg_reward", 0.0),
            trajectory_data=session.get("trajectory", {}),
        )

    async def close(
        self,
        request: Request,
        body: AtroposCloseRequest,
    ) -> AtroposCloseResponse:
        self._sessions.pop(body.env_id, None)

        return AtroposCloseResponse(
            message="Session closed",
            success=True,
        )


if __name__ == "__main__":
    AtroposResourcesServer.run_webserver()
