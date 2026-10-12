"""CPU checks for source formats before any bulk weight export."""
import unittest
import numpy as np
import torch

from modelopt_source import fp8_blocks
from pack_modelopt_experts import repack, verify_repack


class TensorSource:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values[name]


class PackingTests(unittest.TestCase):
    def test_nvfp4_codes_scales_and_global_multiplier(self):
        codes = np.tile(np.arange(16, dtype=np.uint8), (3, 8))
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales = torch.tensor([[.5, 1, 2, 4, 8, 16, 32, 64]] * 3).to(torch.float8_e4m3fn)
        source = TensorSource({'test.weight': torch.from_numpy(packed), 'test.weight_scale': scales,
                               'test.weight_scale_2': torch.tensor(.125)})
        blocks, multiplier = repack(source, 'test')
        self.assertEqual(multiplier, .125)
        report = verify_repack(source, 'test', blocks, multiplier)
        self.assertTrue(report['codes_exact'])
        self.assertTrue(report['group_scales_exact'])
        self.assertEqual(report['modelopt_scale_relative_rms'], 0)

    def test_reject_nonfinite_nvfp4_global_scale(self):
        source = TensorSource({'test.weight': torch.zeros((1, 32), dtype=torch.uint8),
                               'test.weight_scale': torch.ones((1, 4)).to(torch.float8_e4m3fn),
                               'test.weight_scale_2': torch.tensor(float('inf'))})
        with self.assertRaises(ValueError):
            repack(source, 'test')

    def test_fp8_mtp_block_axes_and_multiplier(self):
        weights = torch.ones((256, 256)).to(torch.float8_e4m3fn)
        scales = torch.tensor([[.25, 1], [2, 4]], dtype=torch.bfloat16)
        restored = fp8_blocks(weights, scales)
        for row, col, value in ((0, 0, .25), (127, 127, .25), (127, 128, 1),
                                (128, 127, 2), (128, 128, 4), (255, 255, 4)):
            self.assertEqual(restored[row, col].item(), value)

    def test_reject_fp8_scale_shape_or_nonpositive_scale(self):
        weights = torch.ones((256, 256)).to(torch.float8_e4m3fn)
        for scales in (torch.ones((1, 2)), torch.tensor([[1., 1.], [0., 1.]])):
            with self.assertRaises(ValueError):
                fp8_blocks(weights, scales)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
