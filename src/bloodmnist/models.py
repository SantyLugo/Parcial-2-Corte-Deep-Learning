"""Configurable CNN used by the five controlled BloodMNIST scenarios."""

from __future__ import annotations

import torch
from torch import nn


class BloodCNN(nn.Module):
    """Three convolution blocks with optional BatchNorm and classifier Dropout."""

    def __init__(
        self,
        num_classes: int = 8,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser al menos 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1).")

        # Tres bloques convolucionales aumentan canales 3→32→64→128.
        channels = (3, 32, 64, 128)
        blocks: list[nn.Module] = []
        # Cada bloque aplica convolución, BatchNorm opcional, activación y reducción espacial.
        for in_channels, out_channels in zip(channels, channels[1:]):
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.extend((nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2)))
        # Agrupar los bloques para aplicar la extracción de características en orden.
        self.features = nn.Sequential(*blocks)
        # Reducir cada mapa de características a un único valor espacial por canal.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Activar Dropout solo en los escenarios configurados con probabilidad positiva.
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Extraer características, resumirlas y producir los logits de las ocho clases.
        features = self.features(images)
        # Convertir los mapas reducidos en un vector por imagen antes del clasificador.
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(pooled))
