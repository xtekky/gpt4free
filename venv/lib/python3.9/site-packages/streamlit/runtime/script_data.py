# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
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

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScriptData:
    """Contains parameters related to running a script."""

    main_script_path: str
    command_line: str
    script_folder: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set some computed values derived from main_script_path.

        The usage of object.__setattr__ is necessary because trying to set
        self.script_folder or self.name normally, even within the __init__ method, will
        explode since we declared this dataclass to be frozen.

        We do this in __post_init__ so that we can use the auto-generated __init__
        method that most dataclasses use.
        """
        main_script_path = os.path.abspath(self.main_script_path)
        script_folder = os.path.dirname(main_script_path)
        object.__setattr__(self, "script_folder", script_folder)

        basename = os.path.basename(main_script_path)
        name = str(os.path.splitext(basename)[0])
        object.__setattr__(self, "name", name)
