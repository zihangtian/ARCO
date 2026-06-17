from .base import BaseEnvironment, BaseState
from .registry import DatasetSpec, get_dataset, register_dataset, registered_datasets

# Import dataset modules so they self-register.
# To add a new dataset: create the module and add one import line here.
from . import hotpotqa  # noqa: F401

# Backward-compat re-exports (prefer get_dataset() in new code)
from .hotpotqa import (  # noqa: F401
    HotpotEnv, HotpotState,
    hotpotqa_reward, hotpotqa_parse_output, hotpotqa_parse_thought_action,
    hotpotqa_load_data,
)
