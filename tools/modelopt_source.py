"""The existing read-only checkpoint mapper, with ModelOpt FP8 block scales."""
from pathlib import Path
import sys

STRATA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_source import Source as CheckpointSource, np, torch


def fp8_blocks(values, scales):
    rows, cols = values.shape
    if tuple(scales.shape) != ((rows + 127) // 128, (cols + 127) // 128):
        raise ValueError('FP8 weight_scale_inv does not match 128x128 blocks')
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Invalid FP8 block scale')
    # This is a dequantization multiplier despite the exported name "scale_inv".
    expanded = scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return values.float() * expanded[:rows, :cols]


class Source(CheckpointSource):
    def values(self, name):
        raw = self.get(name)
        if self.headers[name]['dtype'] == 'F8_E4M3':
            value = fp8_blocks(raw, self.get(name + '_scale_inv'))
        else:
            value = raw.float()
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite source tensor: ' + name)
        return value
