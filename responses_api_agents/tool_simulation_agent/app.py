# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from fastapi import Body

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.tool_simulation.app import ToolSimulationVerifyResponse


class ToolSimulationAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: Optional[int] = None
    max_turns: Optional[int] = None


class ToolSimulationRunRequest(BaseRunRequest):
    pass


class ToolSimulationAgent(SimpleResponsesAPIAgent):
    config: ToolSimulationAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="id",
            created_at=1.0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    async def run(self, body: ToolSimulationRunRequest = Body()) -> ToolSimulationVerifyResponse:
        return ToolSimulationVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=await self.responses(body.responses_create_params),
            reward=0.5,
        )


if __name__ == "__main__":
    ToolSimulationAgent.run_webserver()
