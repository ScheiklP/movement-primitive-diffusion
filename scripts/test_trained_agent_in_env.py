import hydra
import torch
import logging

from omegaconf import DictConfig, OmegaConf
import numpy as np
import random

from movement_primitive_diffusion.utils.setup_helper import setup_agent_and_workspace, parse_wandb_to_hydra_config

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="test_trained_agent_in_env")
def main(cfg: DictConfig) -> None:
    # Load wandb config
    wandb_config = OmegaConf.load(cfg.config)

    # Parse wandb config to hydra config
    hydra_config = parse_wandb_to_hydra_config(wandb_config)

    # Seeds:
    if "seed" in hydra_config:
        torch.manual_seed(hydra_config.seed)
        np.random.seed(hydra_config.seed)
        random.seed(hydra_config.seed)

    # Update config with new values
    hydra_config = OmegaConf.merge(hydra_config, cfg.to_change)

    # Setup agent, and workspace
    agent, workspace = setup_agent_and_workspace(hydra_config)

    # Load the weights
    agent.load_pretrained(cfg.weights)

    test_results = workspace.test_agent(agent, cfg.num_trajectories)

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
