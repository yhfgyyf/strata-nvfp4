"""No SSH/GPU: inject failures around the combined check's first lifecycle mutation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name).resolve()
        self.root = self.source / 'bench'
        build = self.root / 'candidate/build'
        build.mkdir(parents=True)
        (build / 'build.json').write_text('{}')
        self.service = True
        self.actions = []
        self.primary = RuntimeError('injected phase failure')
        self.stop_error = False
        self.restore_error = False
        runner = types.ModuleType('run_cases')
        runner.CONFIG = self.source / 'config.json'
        runner.CONFIG.write_text(json.dumps({'env': {}}))
        runner.SCRIPT = self.source / 'schedule.sh'
        runner.PROFILE = self.source / 'profile.bin'
        runner.METADATA = []
        exe = self.source / 'engine/strata-nvfp4-w4a16-r256'
        exe.parent.mkdir()
        for p in (exe, runner.SCRIPT, runner.PROFILE): p.write_bytes(b'unchanged')
        native = self.source / 'native'
        native.mkdir()
        (native / 'exe').write_bytes(exe.read_bytes())
        runner.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        runner.api = lambda *a, **k: {'activity': {'in_flight': 0}}
        runner.miners = lambda: []
        runner.gpu_pids = lambda: []
        runner.server_owner = lambda *a, **k: types.SimpleNamespace(name='321') if self.service else None
        runner.native = lambda *a: native
        def loaded(**kwargs):
            if self.restore_error: raise RuntimeError('injected verification failure')
            assert self.service
        runner.loaded = loaded
        def phase(*args): raise self.primary
        runner.phase = phase
        runner.save = lambda p, value: Path(p).write_text(json.dumps(value))
        capacity = types.ModuleType('capacity_checks')
        capacity.numerical = lambda *a: None
        modules = patch.dict(sys.modules, run_cases=runner, capacity_checks=capacity)
        modules.start()
        self.addCleanup(modules.stop)
        path = Path(__file__).with_name('run_combined_checks.py')
        spec = importlib.util.spec_from_file_location('combined_fixture', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.SOURCE, self.module.ROOT = self.source, self.root
        self.runner = runner
        self.argv = ['combined', '--build', str(build)]

    def control(self, command, **kwargs):
        action = command[-1]
        self.actions.append(action)
        self.service = action == 'start_strata'
        if action == 'stop_strata' and self.stop_error: raise self.primary
        return types.SimpleNamespace(returncode=0)

    def run_failure(self):
        with patch.object(sys, 'argv', self.argv), patch.object(self.module.subprocess, 'run', self.control):
            with self.assertRaises(type(self.primary)) as caught: self.module.main()
        self.assertIs(caught.exception, self.primary)

    def test_stop_failure_after_shutdown_restores(self):
        self.stop_error = True
        self.run_failure()
        self.assertEqual(self.actions, ['stop_strata', 'start_strata'])
        self.assertTrue(self.service)
        self.assertTrue((self.root / 'combined-checks-1/restored.public.json').is_file())

    def test_interruption_restores_then_preserves_interruption(self):
        self.primary = KeyboardInterrupt('injected interruption')
        self.run_failure()
        self.assertEqual(self.actions, ['stop_strata', 'start_strata'])
        self.assertTrue(self.service)

    def test_verification_and_stderr_failure_preserve_primary(self):
        self.restore_error = True
        broken = types.SimpleNamespace(write=lambda *a: (_ for _ in ()).throw(BrokenPipeError()))
        with patch.object(sys, 'stderr', broken): self.run_failure()
        self.assertTrue(self.service)
        self.assertFalse((self.root / 'combined-checks-1/restored.public.json').exists())


if __name__ == '__main__': unittest.main()
