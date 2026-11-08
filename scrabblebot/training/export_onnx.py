"""
Export trained Benoit PyTorch model to ONNX for production Rust client execution.
"""

from __future__ import annotations
import argparse
import os
import onnx
import torch
import torch.nn as nn

from environment import EVENT_DIM, EVENT_HISTORY_LEN, PRIVATE_DIM
from models import BenoitActorCritic


class BenoitONNXWrapper(nn.Module):
    """
    Deterministic inference wrapper for ONNX export.
    Directly produces actions [batch, 2] and state_value [batch, 1].
    """

    def __init__(self, model: BenoitActorCritic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        private: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method returns a single concatenated tensor [actions(2), value(1)]
        or actions [batch, 2] for maximum runtime compatibility across Rust engines.
        Here we output actions: [batch, 2] where:
          actions[:, 0] = bid_fraction in [0, 1]
          actions[:, 1] = play_word_threshold in [0, 1]
        """
        actions, value = self.model.act_deterministic(private, events)
        # Concatenate into [batch, 3] for clean single-output or return (actions, value)
        return torch.cat([actions, value], dim=-1)


def export_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    encoder_type: str = "transformer",
    opset_version: int = 17,
) -> None:
    """Exports model to ONNX with dynamic batch size."""
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")

    model = BenoitActorCritic(encoder_type=encoder_type).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "encoder_type" in ckpt:
                encoder_type = ckpt["encoder_type"]
        else:
            model.load_state_dict(ckpt)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Exporting initialized model.")

    model.eval()
    wrapper = BenoitONNXWrapper(model)

    # Dummy inputs for tracing
    dummy_private = torch.zeros(1, PRIVATE_DIM, dtype=torch.float32)
    dummy_events = torch.zeros(1, EVENT_HISTORY_LEN, EVENT_DIM, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    print(f"Exporting model to: {output_onnx_path} (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_private, dummy_events),
        output_onnx_path,
        input_names=["private", "events"],
        output_names=["output"],  # [batch, 3]: [bid_fraction, play_threshold, value]
        dynamic_axes={
            "private": {0: "batch"},
            "events": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    # Verify ONNX model integrity
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified successfully! File size: {os.path.getsize(output_onnx_path) / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Benoit model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/benoit_final.pt")
    parser.add_argument("--output", type=str, default="benoit.onnx")
    parser.add_argument("--encoder", type=str, default="transformer", choices=["transformer", "gru"])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        encoder_type=args.encoder,
        opset_version=args.opset,
    )

