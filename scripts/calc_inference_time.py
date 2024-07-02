import hydra
import time
import logging

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from movement_primitive_diffusion.utils.setup_helper import setup_train

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)


CONFIG = "experiments/bimanual_tissue_manipulation/train_prodmp_transformer.yaml"


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG)
def main(cfg: DictConfig) -> None:
    # Setup data and agent
    train_dataloader, _, agent, workspace = setup_train(cfg)
    workspace.close()

    input_dt = cfg.dataset_config.dt
    upsampling_dt = 0.001
    mode = "linear"

    observations = []
    extra_inputs = []

    for batch in tqdm(train_dataloader):
        _, observation, extra_input = agent.process_batch(batch)

        batch_size = list(observation.values())[0].shape[0]

        for i in range(batch_size):
            tmp_obs = {}
            tmp_extra = {}
            for k, v in observation.items():
                tmp_obs[k] = v[i].unsqueeze(0)
            for k, v in extra_input.items():
                tmp_extra[k] = v[i].unsqueeze(0)
            observations.append(tmp_obs)
            extra_inputs.append(tmp_extra)

    # Set the agent's weights to the EMA weights, without storing and loading them in very call to predict()
    agent.use_ema_weights()

    # Prediction speed
    start = time.perf_counter()
    for observation, extra_input in tqdm(zip(observations, extra_inputs), total=len(observations), desc="Inference"):
        _ = agent.predict(observation, extra_input)
    end = time.perf_counter()

    # Prediciton speed with upsampling
    start_upsampled = time.perf_counter()
    for observation, extra_input in tqdm(zip(observations, extra_inputs), total=len(observations), desc="Inference upsampled"):
        _ = agent.predict_upsampled(observation, extra_input, input_dt=input_dt, output_dt=upsampling_dt, mode=mode)
    end_upsampled = time.perf_counter()

    # Restore the original weights
    agent.restore_model_weights()

    inference_time = end - start
    # Total time
    log.info(f"Inference time: {inference_time:.4f} seconds")
    # Prediction time per sample
    log.info(f"Prediction time per sample: {inference_time / len(observations):.4f} seconds")
    # Predicitons per second
    log.info(f"Predictions per second: {len(observations) / inference_time:.4f}")

    inference_time_upsampled = end_upsampled - start_upsampled
    log.info(f"Upsampling factor: {input_dt / upsampling_dt} from {input_dt} to {upsampling_dt} with mode {mode}")
    # Total time
    log.info(f"Inference time upsampled: {inference_time_upsampled:.4f} seconds")
    # Prediction time per sample
    log.info(f"Prediction time per sample upsampled: {inference_time_upsampled / len(observations):.4f} seconds")
    # Predicitons per second
    log.info(f"Predictions per second upsampled: {len(observations) / inference_time_upsampled:.4f}")


if __name__ == "__main__":
    main()
