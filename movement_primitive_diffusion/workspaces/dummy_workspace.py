from typing import List
from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace


class DummyWorkspace(BaseWorkspace):
    def __init__(self, *args, **kwargs):
        pass

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        return {"success_rate": 0.0}

    def get_result_dict_keys(self) -> List[str]:
        return ["success_rate"]

    def close(self) -> None:
        pass
