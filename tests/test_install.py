"""Installation contracts that do not require model weights or a GPU."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from install import config, parser, report_ok


class InstallTests(unittest.TestCase):
    def test_default_runtime_contract(self):
        a = parser().parse_args([])
        c = config(a, Path("/pack"), Path("/mtp"), Path("/vision"))
        self.assertEqual(c["host"], "127.0.0.1")
        self.assertEqual(c["port"], 30001)
        self.assertEqual(c["parallel"], 2)
        self.assertEqual(c["env"]["STRATA_PREFILL_NVFP4"], "fp16")
        self.assertEqual(c["env"]["STRATA_PREFILL_RING"], "256")
        self.assertEqual(c["env"]["STRATA_NVFP4_FUSED_MOE"], "2")
        self.assertEqual(c["env"]["STRATA_PLE_EARLY_UPLOAD"], "1")
        self.assertEqual(c["env"]["STRATA_PLE_RAW_UPLOAD"], "1")
        self.assertEqual(c["env"]["STRATA_PLE_RAW_CACHE_MIB"], "64")
        self.assertNotIn("--vram-reserve-mib", c["args"])
        for key, value in (("--prefill", "8192"), ("--max-context", "262144"), ("--kv", "int8")):
            self.assertEqual(c["args"][c["args"].index(key) + 1], value)
        self.assertTrue(c["vision"]["gpu"])
        self.assertEqual(c["reasoning_loop_recovery"], "recover")

    def test_text_only_and_capacity_overrides(self):
        a = parser().parse_args(["--no-vision", "--context", "32768", "--parallel", "1", "--port", "30113"])
        c = config(a, Path("/pack"), Path("/mtp"), Path("/vision"))
        self.assertNotIn("vision", c)
        self.assertNotIn("--vision", c["args"])
        self.assertEqual(c["parallel"], 1)
        self.assertEqual(c["port"], 30113)

    def test_completed_data_must_match_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            self.assertFalse(report_ok(p, "original"))
            p.write_text(json.dumps(dict(passed=True, source_index_sha256="original")))
            self.assertTrue(report_ok(p, "original"))
            with self.assertRaises(ValueError):
                report_ok(p, "different")
            p.write_text(json.dumps(dict(passed=False, source_index_sha256="original")))
            with self.assertRaises(ValueError):
                report_ok(p, "original")


if __name__ == "__main__":
    unittest.main()
