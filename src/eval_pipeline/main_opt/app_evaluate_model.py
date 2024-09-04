
import hydra
import torch
import wandb
from omegaconf import DictConfig



@hydra.main(config_path="../configs/mlp_opt", config_name="evaluation_opt_results.yaml", version_base=None)
def main_pipeline(cfg: DictConfig) -> None:
    """Main function for the pipeline."""
    debug = True
    with wandb.init(project="kite", job_type="mlp_eval", mode="disabled" if debug else "online"):
        for step in cfg.eval_steps:
            hydra.utils.instantiate(step).run()


import torch.nn as nn

if __name__ == "__main__":
    # plt.rcParams["figure.constrained_layout.use"] = True
    # init_config_store()

    # TODO: this is not nice
    # maybe save directly as torchscript model?
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0):
            super().__init__()

            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size
            self.dropout_prob = dropout_prob

            sizes = [input_size] + hidden_sizes + [output_size]
            self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
            self.dropout = nn.Dropout(dropout_prob)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
                x = self.dropout(x)
            return self.layers[-1](x)

        def clone(self):
            return self

    main_pipeline()
