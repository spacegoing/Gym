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
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class ToolSimulationResourcesServerConfig(BaseResourcesServerConfig):
    user_simulation_model_server: ModelServerRef
    user_simulation_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    tool_execution_simulation_model_server: ModelServerRef
    tool_execution_simulation_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    verification_model_server: ModelServerRef
    verification_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class ToolSimulationUserActionRequest(BaseModel):
    pass


class ToolSimulationUserActionResponse(BaseModel):
    pass


class ToolSimulationToolExecutionRequest(BaseModel):
    pass


class ToolSimulationToolExecutionResponse(BaseModel):
    pass


class ToolSimulationVerifyRequest(BaseVerifyRequest):
    pass


class ToolSimulationVerifyResponse(BaseVerifyResponse):
    pass


class ToolSimulationResourcesServer(SimpleResourcesServer):
    config: ToolSimulationResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/generate_user_action")(self.generate_user_action)
        app.post("/generate_tool_execution_result")(self.generate_tool_execution_result)

        return app

    async def verify(self, body: ToolSimulationVerifyRequest) -> ToolSimulationVerifyResponse:
        return ToolSimulationVerifyResponse(**body.model_dump(), reward=1.0)

    async def generate_user_action(self, body: ToolSimulationUserActionRequest) -> ToolSimulationUserActionResponse:
        return ToolSimulationUserActionResponse()

    async def generate_tool_execution_result(
        self, body: ToolSimulationToolExecutionRequest
    ) -> ToolSimulationToolExecutionResponse:
        return ToolSimulationToolExecutionResponse()


if __name__ == "__main__":
    ToolSimulationResourcesServer.run_webserver()
