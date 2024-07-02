import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from movement_primitive_diffusion.utils.setup_helper import setup_train
from movement_primitive_diffusion.utils.helper import count_parameters

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)


CONFIG = "experiments/bimanual_tissue_manipulation/train_prodmp_transformer.yaml"


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG)
def main(cfg: DictConfig) -> None:
    # Setup data and agent
    _, _, agent, workspace = setup_train(cfg)
    workspace.close()

    # Get number of trainable parameters of encoder
    num_encoder_params = count_parameters(agent.encoder)

    # Get number of trainable parameters of model
    num_model_params = count_parameters(agent.model)

    total_num_params = num_encoder_params + num_model_params

    # Format number of trainable parameters with a comma every 1000
    num_encoder_params = "{:,}".format(num_encoder_params)
    num_model_params = "{:,}".format(num_model_params)
    total_num_params = "{:,}".format(total_num_params)

    # Print
    log.info(f"Number of trainable parameters of encoder: {num_encoder_params}")
    log.info(f"Number of trainable parameters of model: {num_model_params}")
    log.info(f"Total number of trainable parameters: {total_num_params}")


if __name__ == "__main__":
    main()
