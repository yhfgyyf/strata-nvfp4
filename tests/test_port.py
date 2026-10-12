"""Verify the bind address before model loading or any network operation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server


class BindProbe(Exception):
    pass


class PortTests(unittest.TestCase):
    def bound_port(self, config, flags=()):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata.json"
            path.write_text(json.dumps(config))
            with patch.object(sys, "argv", ["serve.server", "--config", str(path), *flags]), \
                    patch.object(server, "Server", side_effect=BindProbe) as bind:
                with self.assertRaises(BindProbe):
                    server.main()
                return bind.call_args.args[0][1]

    def test_configured_port(self):
        self.assertEqual(self.bound_port({"port": 30001}), 30001)

    def test_command_line_overrides_config(self):
        for port in (30113, 0):
            with self.subTest(port=port):
                self.assertEqual(self.bound_port({"port": 30001}, ["--port", str(port)]), port)

    def test_default_without_configured_port(self):
        self.assertEqual(self.bound_port({}), 8095)


if __name__ == "__main__":
    unittest.main()
