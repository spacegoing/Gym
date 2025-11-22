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
from enum import StrEnum
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
)


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Optional[dict[str, Any]]
    return_type: Optional[dict[str, Any]]


class ToolSimulationSource(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL_EXECUTION = "tool_execution"


class ToolSimulationBaseStep(BaseModel):
    responses: list[NeMoGymResponse]


class ToolSimulationActionStep(ToolSimulationBaseStep):
    source: Literal[ToolSimulationSource.USER, ToolSimulationSource.AGENT]
    extracted_content: Annotated[
        Union[NeMoGymEasyInputMessage, NeMoGymResponseFunctionToolCall],
        Field(discriminator="type"),
    ]


class ToolSimulationToolExecutionStep(ToolSimulationBaseStep):
    source: Literal[ToolSimulationSource.TOOL_EXECUTION] = ToolSimulationSource.TOOL_EXECUTION
    extracted_content: NeMoGymFunctionCallOutput


ToolSimulationStep: TypeAlias = Annotated[
    Union[ToolSimulationActionStep, ToolSimulationToolExecutionStep],
    Field(discriminator="source"),
]


class ToolSimulationTrajectory(BaseModel):
    steps: list[ToolSimulationStep]
