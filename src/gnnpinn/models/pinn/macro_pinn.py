"""Macro-scale PINN interface."""

from __future__ import annotations

import math
from typing import Any

from .coordinate_networks import MLP, MLPConfig, ResidualMLP, _activation


def _torch() -> Any:
    import torch

    return torch


class MacroPINN(_torch().nn.Module):
    """Coordinate-based macro field model.

    The first implementation is intentionally minimal: concatenate spatial
    coordinates, time, and optional process/material parameters, then predict
    requested macro fields such as temperature or displacement.
    """

    def __init__(
        self,
        coord_dim: int,
        field_dim: int,
        param_dim: int = 0,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        activation: str = "tanh",
        conditioning_mode: str = "concat",
        film_strength: float = 1.0,
        route_film_prior: float = 0.5,
        route_trainable: bool = True,
        spacetime_encoding: str = "raw",
        spacetime_fourier_bands: int = 4,
        backbone_mode: str = "mlp",
        backbone_residual_scale: float = 1.0,
        process_encoder_mode: str = "none",
        process_encoder_dim: int = 0,
    ):
        super().__init__()
        torch = _torch()
        normalized_mode = conditioning_mode.lower()
        if normalized_mode not in {"concat", "film", "concat_film", "routed"}:
            raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")
        if normalized_mode in {"film", "concat_film", "routed"} and param_dim <= 0:
            raise ValueError(f"{normalized_mode} conditioning requires param_dim > 0")
        if normalized_mode == "routed" and not 0.0 < route_film_prior < 1.0:
            raise ValueError("route_film_prior must be between 0 and 1")
        normalized_spacetime_encoding = spacetime_encoding.lower()
        if normalized_spacetime_encoding not in {"raw", "fourier"}:
            raise ValueError(f"Unsupported spacetime encoding: {spacetime_encoding}")
        if spacetime_fourier_bands < 1:
            raise ValueError("spacetime_fourier_bands must be >= 1")
        normalized_backbone_mode = backbone_mode.lower()
        if normalized_backbone_mode not in {"mlp", "residual"}:
            raise ValueError(f"Unsupported backbone mode: {backbone_mode}")
        if backbone_residual_scale < 0.0:
            raise ValueError("backbone_residual_scale must be non-negative")
        if normalized_backbone_mode == "residual" and num_hidden_layers < 1:
            raise ValueError("residual backbone requires num_hidden_layers >= 1")
        normalized_process_encoder_mode = process_encoder_mode.lower()
        if normalized_process_encoder_mode not in {"none", "linear"}:
            raise ValueError(f"Unsupported process_encoder_mode: {process_encoder_mode}")
        if normalized_process_encoder_mode != "none" and param_dim <= 0:
            raise ValueError("process encoder requires param_dim > 0")
        encoded_param_dim = param_dim
        if normalized_process_encoder_mode != "none":
            if process_encoder_dim < 0:
                raise ValueError("process_encoder_dim must be non-negative")
            encoded_param_dim = process_encoder_dim or param_dim
            if encoded_param_dim <= 0:
                raise ValueError("process_encoder_dim must be positive when process encoder is enabled")
        self.coord_dim = coord_dim
        self.field_dim = field_dim
        self.raw_param_dim = param_dim
        self.param_dim = encoded_param_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.conditioning_mode = normalized_mode
        self.film_strength = float(film_strength)
        self.route_film_prior = float(route_film_prior)
        self.route_trainable = bool(route_trainable)
        self.spacetime_encoding = normalized_spacetime_encoding
        self.spacetime_fourier_bands = int(spacetime_fourier_bands)
        self.backbone_mode = normalized_backbone_mode
        self.backbone_residual_scale = float(backbone_residual_scale)
        self.process_encoder_mode = normalized_process_encoder_mode
        self.process_encoder_dim = int(encoded_param_dim)
        self.process_encoder_identity_initialized = False
        self.process_encoder = None
        if self.process_encoder_mode == "linear":
            self.process_encoder = torch.nn.Linear(param_dim, encoded_param_dim)
            torch.nn.init.zeros_(self.process_encoder.weight)
            torch.nn.init.zeros_(self.process_encoder.bias)
            copy_dim = min(param_dim, encoded_param_dim)
            with torch.no_grad():
                for index in range(copy_dim):
                    self.process_encoder.weight[index, index] = 1.0
            self.process_encoder_identity_initialized = True
        spacetime_dim = self._spacetime_feature_dim()
        self.spacetime_dim = spacetime_dim
        if self.conditioning_mode == "concat":
            self.backbone = self._build_backbone(
                input_dim=spacetime_dim + encoded_param_dim,
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
            )
            return

        if self.conditioning_mode == "routed":
            self.concat_expert = self._build_backbone(
                input_dim=spacetime_dim + encoded_param_dim,
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
            )
            self.route_gate = torch.nn.Linear(encoded_param_dim, 1)
            torch.nn.init.zeros_(self.route_gate.weight)
            torch.nn.init.constant_(self.route_gate.bias, math.log(route_film_prior / (1.0 - route_film_prior)))
            if not self.route_trainable:
                for parameter in self.route_gate.parameters():
                    parameter.requires_grad_(False)

        self.activation = _activation(activation)
        self.hidden_layers = torch.nn.ModuleList()
        self.film_generators = torch.nn.ModuleList()
        in_dim = spacetime_dim
        if self.conditioning_mode == "concat_film":
            in_dim += encoded_param_dim
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            generator = torch.nn.Linear(encoded_param_dim, 2 * hidden_dim)
            torch.nn.init.zeros_(generator.weight)
            torch.nn.init.zeros_(generator.bias)
            self.film_generators.append(generator)
            in_dim = hidden_dim
        self.output_layer = torch.nn.Linear(in_dim, field_dim)

    def _build_backbone(
        self,
        *,
        input_dim: int,
        field_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
    ) -> Any:
        config = MLPConfig(
            input_dim=input_dim,
            output_dim=field_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )
        if self.backbone_mode == "residual":
            return ResidualMLP(config, residual_scale=self.backbone_residual_scale)
        return MLP(config)

    def _spacetime_feature_dim(self) -> int:
        base_dim = self.coord_dim + 1
        if self.spacetime_encoding == "raw":
            return base_dim
        return base_dim * (1 + 2 * self.spacetime_fourier_bands)

    def _spacetime_features(self, coords: Any, time: Any) -> Any:
        torch = _torch()
        if time.ndim == 1:
            time = time[:, None]
        base = torch.cat([coords, time], dim=-1)
        if self.spacetime_encoding == "raw":
            return base
        features = [base]
        for band in range(self.spacetime_fourier_bands):
            angle = math.pi * (2.0**band) * base
            features.append(torch.sin(angle))
            features.append(torch.cos(angle))
        return torch.cat(features, dim=-1)

    def _encode_params(self, params: Any | None) -> Any | None:
        if params is None:
            return None
        if self.process_encoder is None:
            return params
        return self.process_encoder(params)

    def forward(self, coords: Any, time: Any, params: Any | None = None) -> Any:
        torch = _torch()
        spacetime = self._spacetime_features(coords, time)
        encoded_params = self._encode_params(params)
        if self.conditioning_mode in {"film", "concat_film", "routed"}:
            if encoded_params is None:
                raise ValueError(f"params are required when conditioning_mode={self.conditioning_mode!r}")
            concat_output = None
            if self.conditioning_mode == "routed":
                concat_output = self.concat_expert(torch.cat([spacetime, encoded_params], dim=-1))
            hidden_inputs = [spacetime]
            if self.conditioning_mode == "concat_film":
                hidden_inputs.append(encoded_params)
            hidden = torch.cat(hidden_inputs, dim=-1)
            for layer_index, (layer, generator) in enumerate(zip(self.hidden_layers, self.film_generators)):
                previous_hidden = hidden
                hidden = self.activation(layer(hidden))
                gamma, beta = generator(encoded_params).chunk(2, dim=-1)
                hidden = hidden * (1.0 + self.film_strength * gamma) + self.film_strength * beta
                if self.backbone_mode == "residual" and layer_index > 0:
                    hidden = previous_hidden + self.backbone_residual_scale * hidden
            film_output = self.output_layer(hidden)
            if self.conditioning_mode == "routed":
                gate = self.routing_gate(params)
                return (1.0 - gate) * concat_output + gate * film_output
            return film_output
        if self.param_dim:
            if encoded_params is None:
                raise ValueError("params are required when param_dim > 0")
            return self.backbone(torch.cat([spacetime, encoded_params], dim=-1))
        return self.backbone(spacetime)

    def routing_gate(self, params: Any) -> Any:
        """Return the FiLM-expert weight for routed conditioning."""

        if self.conditioning_mode != "routed":
            raise ValueError("routing_gate is only available for conditioning_mode='routed'")
        torch = _torch()
        encoded_params = self._encode_params(params)
        return torch.sigmoid(self.route_gate(encoded_params))

    def routing_summary(self, params: Any) -> dict[str, Any] | None:
        """Summarize routed conditioning gate values for run metadata."""

        if self.conditioning_mode != "routed":
            return None
        gate = self.routing_gate(params).detach().cpu().reshape(-1)
        return {
            "enabled": True,
            "film_prior": self.route_film_prior,
            "trainable": self.route_trainable,
            "film_gate_mean": float(gate.mean()),
            "film_gate_std": float(gate.std(unbiased=False)),
            "film_gate_min": float(gate.min()),
            "film_gate_max": float(gate.max()),
        }
