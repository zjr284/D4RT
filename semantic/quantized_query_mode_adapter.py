"""Match the frozen R105 training boundary without altering geometry or weights."""
import torch
from semantic.query_mode_adapter import QueryModeSemanticD4RT


class QuantizedQueryModeSemanticD4RT(QueryModeSemanticD4RT):
    """CUDA BF16 base/autocast; FP32 semantic-tail parameters, BF16 tail input."""

    def source_prefix(self, video, query, memory):
        # Residual prefix outputs remain FP32 under autocast. R105 load_features
        # explicitly rounds them to BF16 before training and cached evaluation.
        return super().source_prefix(video, query, memory).to(torch.bfloat16)
