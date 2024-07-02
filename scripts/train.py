import hydra
import torch
import wandb
import logging
import operator

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

from movement_primitive_diffusion.utils.helper import dictionary_to_device, format_loss
from movement_primitive_diffusion.utils.setup_helper import setup_train, setup_wandb_metrics, get_group_from_override
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)


CONFIG = "experiments/bimanual_tissue_manipulation/train_prodmp_transformer.yaml"


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG)
def main(cfg: DictConfig) -> float:
    # Seeds:
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # Set the performance comparison operator that is used to determine whether a model is better than the previous best
    performance_comparison_operator = operator.ge if cfg.performance_direction == "max" else operator.le
    performance_metric = cfg.performance_metric

    # Setup data, agent, and workspace
    train_dataloader, val_dataloader, agent, workspace = setup_train(cfg)

    # Initialize wandb
    # Init wandb stored config with entire hydra config used in this experiment
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if "group_from_overrides" in cfg and cfg.group_from_overrides:
        cfg.wandb.group = get_group_from_override()

    wandb_kwargs = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "group": cfg.wandb.group,
        "mode": cfg.wandb.mode,
        "config": wandb.config,
    }

    if "run_name" in cfg.wandb:
        wandb_kwargs["name"] = cfg.wandb.run_name
    elif "name_from_overrides" in cfg and cfg.name_from_overrides:
        wandb_kwargs["name"] = get_group_from_override(ignore_keys=cfg.get("ignore_in_name", []))

    # init wandb logger and config from hydra path
    wandb_run = wandb.init(**wandb_kwargs)

    # Check where to log the model's state dict
    if cfg.wandb.mode == "disabled":
        raise NotImplementedError("Logging to file is not implemented yet if wandb is disabled.")
        # TODO
        # Technically, we could set hydra.run.dir, but that does not appear in cfg.
        # We could also set hydra.job.chdir, and get the path with os.getcwd(), but that also creates the wandb folder within the dir.
    else:
        logging_path = Path(wandb_run.dir)

    # Setup wandb metrics
    workspace_result_keys = workspace.get_result_dict_keys()
    setup_wandb_metrics(workspace_result_keys, performance_metric)

    best_performance_value = -torch.inf if cfg.performance_direction == "max" else torch.inf
    best_epoch = 0
    current_early_stopping_patience = 0
    done = False

    if not cfg.early_stopping and cfg.epochs is None:
        raise ValueError("Either early stopping or epochs must be set otherwise training will never stop.")

    epoch_magnitude = len(str(cfg.epochs))
    with tqdm(range(cfg.epochs)) as pbar_epochs:
        for current_epoch in pbar_epochs:
            train_losses = []
            val_losses = []
            start_point_deviations = []
            end_point_deviations = []

            epoch_string = str(current_epoch).zfill(epoch_magnitude)

            with tqdm(train_dataloader, leave=False) as pbar_train:
                # Train for one epoch
                for batch in pbar_train:
                    if not cfg.dataset_fully_on_gpu:
                        batch = dictionary_to_device(batch, cfg.device)
                    loss_value = agent.train_step(batch)
                    train_losses.append(loss_value)

                    pbar_train.set_description(f"Train epoch {epoch_string}/{cfg.epochs}")
                    pbar_train.set_postfix(loss=format_loss(loss_value))

            # Validate for one epoch
            with tqdm(val_dataloader, leave=False) as pbar_val:
                for batch in pbar_val:
                    if not cfg.dataset_fully_on_gpu:
                        batch = dictionary_to_device(batch, cfg.device)
                    val_loss_value, start_point_deviation, end_point_deviation = agent.evaluate(batch)
                    val_losses.append(val_loss_value)
                    start_point_deviations.append(start_point_deviation)
                    end_point_deviations.append(end_point_deviation)

                    pbar_val.set_description(f"Valid epoch {epoch_string}/{cfg.epochs}")
                    pbar_val.set_postfix(val_loss=format_loss(val_loss_value))

            # Log the epoch info
            mean_train_loss = sum(train_losses) / len(train_losses)
            mean_val_loss = sum(val_losses) / len(val_losses)
            mean_start_point_deviation = sum(start_point_deviations) / len(start_point_deviations)
            mean_end_point_deviation = sum(end_point_deviations) / len(end_point_deviations)
            epoch_info = {
                "epoch": current_epoch,
                "loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "start_point_deviation": mean_start_point_deviation,
                "end_point_deviation": mean_end_point_deviation,
                "lr": agent.optimizer.param_groups[0]["lr"],
            }
            wandb.log(epoch_info)

            # Test agent in workspace
            if current_epoch % cfg.eval_in_env_after_epochs == 0:
                test_results = workspace.test_agent(agent, cfg.num_trajectories_in_env)
                wandb.log(test_results)
                pbar_epochs.set_description(f"Epoch {epoch_string}/{cfg.epochs}")
                pbar_epochs.set_postfix(**test_results)
            else:
                test_results = {}

            # Check if the current model is better than the previous best
            combined_epoch_info = {**epoch_info, **test_results}
            # NOTE: We do this check, because the performance_metric could be either in epoch_info or test_results.
            # If there are no test results in this epoch, we do not want to trigger early stopping based on the epoch_info.
            if performance_metric in combined_epoch_info:
                if performance_comparison_operator(combined_epoch_info[performance_metric], best_performance_value):
                    best_performance_value = combined_epoch_info[performance_metric]
                    best_epoch = current_epoch
                    current_early_stopping_patience = 0
                    # Write info about the best model to a text file
                    with open(logging_path / "best_model_info.txt", "w") as f:
                        f.write(f"epoch={epoch_string}, {performance_metric=}, {best_performance_value=}, {mean_train_loss=}, {mean_val_loss=}\n")
                    # Overwrite the best model
                    agent.save_model(logging_path / "best_model.pth")
                else:
                    early_stopping_warmup_epochs = cfg.get("early_stopping_warmup_epochs", None)
                    if early_stopping_warmup_epochs is not None and current_epoch >= early_stopping_warmup_epochs:
                        current_early_stopping_patience += 1
                        # Check if training should be stopped due to early stopping
                        early_stopping = cfg.get("early_stopping", False)
                        if early_stopping and current_early_stopping_patience >= cfg.early_stopping_patience:
                            done = True
                            log.log(logging.INFO, f"Early stopping after {current_epoch} epochs with best {performance_metric} of {best_performance_value} at epoch {best_epoch}.")

            # Save intermediate checkpoints of the model
            if cfg.save_distance is not None and current_epoch % cfg.save_distance == 0:
                agent.save_model(logging_path / f"model_epoch_{epoch_string}.pth")

            # Log the current best model metrics
            wandb.log({f"best_{performance_metric}": best_performance_value})
            wandb.log({"best_epoch": best_epoch})

            # Check if training should be stopped due to reaching the maximum number of epochs
            if cfg.epochs is not None and current_epoch >= cfg.epochs or done:
                log.log(logging.INFO, f"Finished after {current_epoch} epochs with best {performance_metric} of {best_performance_value} at epoch {best_epoch}.")
                break

    # Save the final model
    # NOTE: This ist not the best model, but the model of the final epoch
    model_name = "model_last_epoch.pth"
    agent.save_model(logging_path / model_name)

    # Close the workspace and it's environments.
    # If that takes longer than 60 seconds, terminate the subproccesses of the vectorized environment.
    if isinstance(workspace, BaseVectorWorkspace):
        workspace.close(timeout=60)
    else:
        workspace.close()

    # Finish the wandb run
    wandb_run.finish()

    return best_performance_value


if __name__ == "__main__":
    main()
