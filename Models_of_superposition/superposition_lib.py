import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple, Dict

from tqdm.notebook import tqdm
from dataclasses import dataclass

from scipy.optimize import fsolve


device = t.device("cuda" if t.cuda.is_available() else "cpu")


def get_angle(d, n):
    def equation(x):
        return d - (np.cos(x) - np.sin(x) / np.sqrt(n)) ** 2

    # Initial guess for x
    x0 = 0

    # Solve the equation numerically
    solution = fsolve(equation, x0)

    return solution[0]


def draw_samples(p, num_samples, max_length):
    q = 1 - p
    if q == 0:
        samples = t.ones((num_samples,)) * t.inf
    else:
        dist = t.distributions.Geometric(q)
        samples = dist.sample((num_samples,))
    samples = t.clamp(samples.long(), max=max_length)
    return samples


def generate_feature_mask_group(feat_is_present, p_transfer):
    b, i, f = feat_is_present.shape
    # flatten batch and instance dimensions
    feat_is_present_flat = einops.rearrange(feat_is_present, "b i f -> (b i) f")

    # Random permutations for each row
    permutations = t.argsort(t.rand(b * i, f), dim=1).to(device)

    # Number of active features for each instance
    active_feature_number = draw_samples(p_transfer, b * i, f)

    # Generate masks where first few elements are True based on active_feature_number
    mask = t.arange(f).repeat(b * i, 1) < active_feature_number.unsqueeze(1)
    mask = mask.to(device)

    # Apply permutations to mask
    permuted_mask = t.zeros_like(mask).to(device)
    permuted_mask.scatter_(1, permutations, mask).to(device)

    # Applying mask to feat_is_present_flat
    # Copying the first active feature across each selected feature position
    first_feature_values = feat_is_present_flat.gather(
        1, permutations[:, :1].repeat(1, f)
    )
    feat_is_present_flat = t.where(
        permuted_mask,
        first_feature_values,
        t.tensor(0.0, device=feat_is_present.device),
    )

    # Reshape to original shape
    feat_is_present = einops.rearrange(
        feat_is_present_flat, "(b i) f -> b i f", b=b, i=i
    )
    # convert to boolean
    feat_is_present = feat_is_present > 0.5

    return feat_is_present


def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


def create_diagonal_mask(n_features: int) -> t.Tensor:
    return t.eye(n_features, dtype=t.bool)


def create_group_mask(n_features: int, group_size: int) -> t.Tensor:
    mask = t.zeros((n_features, n_features), dtype=t.bool)
    for group_number in range(0, n_features, group_size):
        for i in range(group_size):
            for j in range(group_size):
                mask[group_number + i, group_number + j] = True
    return mask & ~create_diagonal_mask(n_features)


def create_others_mask(n_features: int, group_size: int) -> t.Tensor:
    return ~create_group_mask(n_features, group_size) & ~create_diagonal_mask(
        n_features
    )


def create_mask_from_group_members(
    n_features: int, group_members: List[int]
) -> t.Tensor:
    mask = t.zeros((n_features, n_features), dtype=t.bool)
    for i in group_members:
        for j in group_members:
            mask[i, j] = True
    return mask


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device=device,
        groupings: Optional[Union[List[Dict], List[List[Dict]]]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.groupings = groupings

        self.accuracy_matrix = t.diag(t.ones(cfg.n_features)).to(device)
        self.accuracy_matrix = t.stack([self.accuracy_matrix] * cfg.n_instances)
        if groupings is not None:
            for instance, instance_groups in enumerate(groupings):
                for group_dict in instance_groups:
                    grou_members = group_dict["members"]
                    semantic_distance = group_dict["semantic_distance"]
                    group_mask = create_mask_from_group_members(
                        cfg.n_features, grou_members
                    ).to(device)
                    diag_mask = create_diagonal_mask(cfg.n_features).to(device)
                    n_group = len(grou_members)
                    angle = get_angle(semantic_distance, n_group)

                    self.accuracy_matrix[instance] = t.where(
                        group_mask,
                        np.sin(angle) / n_group,
                        self.accuracy_matrix[instance],
                    )
                    self.accuracy_matrix[instance] = t.where(
                        group_mask & diag_mask,
                        np.cos(angle),
                        self.accuracy_matrix[instance],
                    )
        if feature_probability is None:
            feature_probability = t.ones(())
        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_instances, cfg.n_features)
        )
        if importance is None:
            importance = t.ones(())
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to(
            (cfg.n_instances, cfg.n_features)
        )
        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))
            )
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)

    def encode(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances hidden"]:
        hidden_layer = t.einsum("...if,ihf->...ih", features, self.W)
        return F.relu(hidden_layer)

    def decode(
        self, hidden: Float[Tensor, "... instances hidden"]
    ) -> Float[Tensor, "... instances features"]:
        return t.einsum("...ih,ihf->...if", hidden, self.W) + self.b_final

    def forward(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = self.encode(features)
        reconstructed = self.decode(hidden)
        return reconstructed

    def generate_batch(
        self, batch_size
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of data. We'll return to this function later when we apply correlations.
        """
        # Generate the features, before randomly setting some to zero
        feat = t.rand(
            (batch_size, self.cfg.n_instances, self.cfg.n_features),
            device=self.W.device,
        )
        # Generate a random boolean array, which is 1 wherever we'll keep a feature, and zero where we'll set it to zero
        feat_seeds = t.rand(
            (batch_size, self.cfg.n_instances, self.cfg.n_features),
            device=self.W.device,
        )
        feat_is_present = feat_seeds <= self.feature_probability
        # Create our batch from the features, where we set some to zero

        if self.groupings is not None:
            if isinstance(self.groupings[0], list):
                for instance, instance_groups in enumerate(self.groupings):
                    for group_dict in instance_groups:
                        group_members = group_dict["members"]
                        group_p_transfer = group_dict["p_transfer"]

                        lst = t.tensor(group_members)

                        feat_is_present[:, instance : instance + 1, lst] = (
                            generate_feature_mask_group(
                                feat_is_present[:, instance : instance + 1, lst],
                                group_p_transfer,
                            )
                        )
            else:
                for group_dict in self.groupings:
                    group_members = group_dict["members"]
                    group_p_transfer = group_dict["p_transfer"]

                    lst = t.tensor(group_members)

                    feat_is_present[:, :, lst] = generate_feature_mask_group(
                        feat_is_present[:, :, lst], group_p_transfer
                    )

        batch = t.where(feat_is_present, feat, 0.0)
        return batch

    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `model.importance` will always have shape (n_instances, n_features).
        """

        diff = out - batch
        rotated_diff = t.einsum("bif,ifn->bin", diff, self.accuracy_matrix)
        total_diff = (rotated_diff**2 * self.importance).sum(dim=-1)
        return total_diff.mean()

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / self.cfg.n_instances, lr=step_lr
                )

    def select(model, choices):
        if isinstance(choices, int):
            choices = [choices]

        cfg = model.cfg
        cfg.n_instances = len(choices)
        feature_probability = model.feature_probability[choices]
        groupings = [model.groupings[choice] for choice in choices]
        selected_model = Model(
            cfg=cfg, feature_probability=feature_probability, groupings=groupings
        )
        selected_model.W = t.nn.Parameter(model.W[choices])
        selected_model.b_final = t.nn.Parameter(model.b_final[choices])

        return selected_model


def WtW(weights: t.Tensor) -> t.Tensor:
    return einops.einsum(
        weights,
        weights,
        "instances hidden feats_i, instances hidden feats_j -> instances feats_i feats_j",
    )
