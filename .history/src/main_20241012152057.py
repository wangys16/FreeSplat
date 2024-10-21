import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

import argparse

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)



def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    if cfg.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
        print(cyan(f"Saving outputs to {output_dir}."))
        latest_run = output_dir.parents[1] / "latest-run"
    else:
        output_dir = Path(f'outputs/{cfg.output_dir}')
        if not os.path.exists(f'outputs/{cfg.output_dir}'):
            os.mkdir(f'outputs/{cfg.output_dir}')
        print(cyan(f"Saving outputs to {output_dir}."))
        latest_run = Path("outputs") / "latest-run"


    run = wandb.init(dir=output_dir)
    
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            experiment=run
        )
        callbacks.append(LearningRateMonitor("step", True))

    else:
        logger = LocalLogger()

    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy='auto'
        if torch.cuda.device_count() > 1
        else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        check_val_every_n_epoch=None,
    )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, 
                        depth_range=[cfg.dataset.near, cfg.dataset.far])
    cfg.test.output_path = output_dir

    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)
    
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        data_module,
        cfg_dict=cfg_dict,
        run_dir=run.dir,
        num_context_views=cfg.dataset.view_sampler.num_context_views,
        dataset_name=cfg.dataset.name,
    )


    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
