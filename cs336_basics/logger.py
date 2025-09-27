import html
from datetime import datetime
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import swanlab
import wandb
from torch.utils.tensorboard.writer import SummaryWriter

class Logger:
    """
    A unified logger that supports TensorBoard, Weights & Biases (wandb), and SwanLab.
    It initializes based on the configuration provided and allows logging of metrics and text.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger_type = config.logger.type
        self.writer = None

        config_dict = OmegaConf.to_container(config, resolve=True)

        if self.logger_type == "swanlab":
            swanlab.init(
                project=config.logger.project_name,
                name=config.logger.run_name,
                config=config_dict,
                reinit=True
            )
            self.writer = swanlab
        elif self.logger_type == "wandb":
            wandb.init(
                project=config.logger.project_name,
                name=config.logger.run_name,
                config=config_dict,
                reinit=True
            )
            self.writer = wandb
        elif self.logger_type == "tensorboard":
            current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
            log_dir = Path.cwd() / "tensorboard" / current_time
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            print(f"Unknown logger type '{self.logger_type}'. Logging will be disabled.")
            self.logger_type = "none"

    def log_metrics(self, metrics: dict, step: int):
        if self.logger_type == "none":
            return
        if self.logger_type in ["wandb", "swanlab"]:
            self.writer.log(metrics, step=step)
        elif self.logger_type == "tensorboard":
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_text(self, key: str, text: str, step: int):
        if self.logger_type == "none":
            return
        if self.logger_type == "wandb":
            # wandb can directly record HTML
            wandb.log({key: wandb.Html(text)}, step=step)
        elif self.logger_type == "swanlab":
            swanlab.log({key: swanlab.Text(text)}, step=step)
        elif self.logger_type == "tensorboard":
            # TensorBoard treats text as markdown
            escaped_text = html.escape(text)
            self.writer.add_text(key, f"<pre>{escaped_text}</pre>", step)

    def log_table(self, key: str, table: dict):
        if self.logger_type == "wandb":
            wandb.log({key: wandb.Table(dataframe=pd.DataFrame(table))})
        elif self.logger_type == "swanlab":
            etable = swanlab.echarts.Table()
            etable.add(
                list(table.keys()),
                [list(row) for row in zip(*table.values())]  # Transpose the table for swanlab
            )
            swanlab.log({"completions": etable})
        elif self.logger_type == "tensorboard":
            # TensorBoard does not support direct table logging, so we convert to text
            table_str = pd.DataFrame(table).to_html(index=False)
            self.writer.add_text(key, f"<pre>{html.escape(table_str)}</pre>")

    def close(self):
        if self.writer and self.logger_type in ["wandb", "swanlab"]:
            self.writer.finish()
        elif self.writer and self.logger_type == "tensorboard":
            self.writer.close()