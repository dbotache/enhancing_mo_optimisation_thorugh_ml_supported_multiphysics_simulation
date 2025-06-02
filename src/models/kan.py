import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    """

    KANLinear is a kernel-based adaptive linear layer that combines
    standard linear transformations with learned B-spline interpolations.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        grid_size (int): Number of bins in the B-spline grid.
        spline_order (int): Order of the B-spline (e.g., 3 for cubic).
        scale_noise (float): Scale of noise added to spline initialization.
        scale_base (float): Scaling factor for the base linear weights.
        scale_spline (float): Scaling factor for spline weights.
        enable_standalone_scale_spline (bool): Whether to use per-weight spline scalers.
        base_activation (nn.Module): Activation function applied before base linear transformation.
        grid_eps (float): Interpolation factor between uniform and data-driven grid.
        grid_range (list): Initial range for spline grid (e.g., [-1, 1]).

    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()

        """
        if (grid_size + 1) < (grid_size + spline_order):
            print(f'---------Grid_size: {grid_size}, spline_order: {spline_order}-----------')
            raise ValueError(
                f"Invalid configuration: grid_size + 1 ({grid_size + 1}) must be >= grid_size + spline_order ({grid_size + spline_order}). "
                "Consider using a smaller spline_order or a larger grid_size."
            )
        """


        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize the B-spline grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Parameters for linear and spline weights
        self.base_weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        # Optional standalone spline scaling
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.empty(out_features, in_features))

        # Hyperparameters
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
                ) * self.scale_noise / self.grid_size
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input.

        Args:
            x (torch.Tensor): Input of shape (batch_size, in_features)

        Returns:
            torch.Tensor: B-spline basis of shape (batch_size, in_features, grid_size + spline_order)
        """
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, k + 1 :] - x)
                / (self.grid[:, k + 1 :] - self.grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Solve for B-spline coefficients given input-output examples.

        Args:
            x (torch.Tensor): Input grid, shape (grid_size+1, in_features)
            y (torch.Tensor): Function values, shape (grid_size+1, in_features, out_features)

        Returns:
            torch.Tensor: Coefficient tensor, shape (out_features, in_features, grid_size + spline_order)
        """
        A = self.b_splines(x).transpose(0, 1)  # (in_features, m, n)
        B = y.transpose(0, 1)  # (in_features, m, out_features)
        A_T = A.transpose(1, 2)  # (in_features, n, m)

        # Regularized least squares: (A^T A + λI)^-1 A^T B
        lambda_reg = 1e-4
        ATA = torch.bmm(A_T, A) + lambda_reg * torch.eye(A.shape[2], device=A.device).unsqueeze(0)
        ATB = torch.bmm(A_T, B)
        coeffs = torch.linalg.solve(ATA, ATB)  # (in_features, n, out_features)

        return coeffs.permute(2, 0, 1).contiguous()  # (out_features, in_features, basis)


    @property
    def scaled_spline_weight(self):
        """Apply spline scaling if enabled."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        """Apply linear and spline transformations."""
        x_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1)
        )
        output = base_output + spline_output
        return output.view(*x_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """Update spline grid based on data distribution."""
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=x.device).unsqueeze(1) * uniform_step
            + x_sorted[0] - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([
            grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)
        ], dim=0)

        self.grid.copy_(grid.T)
        self.spline_weight.copy_(self.curve2coeff(x, spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute L1 and entropy-based regularization on spline weights.

        Returns:
            torch.Tensor: Regularization loss
        """
        l1 = self.spline_weight.abs().mean(-1)
        act_loss = l1.sum()
        p = l1 / (act_loss + 1e-8)
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        return regularize_activation * act_loss + regularize_entropy * entropy


class KAN(torch.nn.Module):
    """
        KAN (Kernel-based Adaptive Network) is a multilayer architecture composed
        of KANLinear layers, each combining linear weights with learned spline-based
        nonlinearities for flexible function approximation.

        Args:
            layers_hidden (list): List of layer widths. Must include input and output sizes.
            grid_size (int): Number of bins in the B-spline grid per layer.
            spline_order (int): Order of the B-spline (e.g., 3 for cubic splines).
            scale_noise (float): Scale of noise for spline initialization.
            scale_base (float): Scale for linear (base) weights.
            scale_spline (float): Scale for spline weights.
            base_activation (nn.Module): Activation used before linear transformation.
            grid_eps (float): Blend factor between uniform and adaptive grid update.
            grid_range (list): Initial range for the B-spline grid.
        """

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """Forward pass with optional grid update."""
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Aggregate regularization loss from all layers."""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss = 0.0
    model.train()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        output = model(X_batch.float())
        loss = loss_fn(output, y_batch.view(-1, y_batch.shape[-1]).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = 0.0
    model.eval()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch.float())
        loss = loss_fn(output, y_batch.view(-1, y_batch.shape[-1]).float())
        valid_loss += loss.item()

    return valid_loss