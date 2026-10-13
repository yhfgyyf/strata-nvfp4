import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('run_matrix.py'))
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def base(self):
        return dict(args=['--pcie-frac','0','--max-context','262144','--prefill','8192'],
            env={'STRATA_PLE_EARLY_UPLOAD':'1','STRATA_PLE_RAW_UPLOAD':'1','STRATA_PLE_RAW_CACHE_MIB':'64'},parallel=2)

    def test_modes_are_distinct_and_base_not_mutated(self):
        base=self.base(); before=json.dumps(base)
        for mode in ('kernel','dma'):
            for frac in (0,.1,.2):
                cfg=matrix.candidate_config(base,dict(mode=mode,frac=frac),Path('/tmp'),'case')
                self.assertEqual(cfg['args'][cfg['args'].index('--pcie-mode')+1],mode)
                self.assertEqual(cfg['args'][cfg['args'].index('--pcie-frac')+1],str(frac))
                self.assertEqual(cfg['env']['STRATA_PLE_RAW_CACHE_MIB'],'64')
                self.assertEqual(cfg['env']['STRATA_PREFILL_RING'],'256')
                self.assertEqual(cfg['parallel'],2)
        self.assertEqual(json.dumps(base),before)

    def test_single_lag_and_async_are_separate(self):
        base=self.base();out=Path('/tmp')
        lag=matrix.candidate_config(base,dict(mode='kernel',frac=0,parallel=1,lag=2),out,'lag')
        self.assertEqual(lag['parallel'],1)
        self.assertEqual(lag['args'][lag['args'].index('--batch')+1],'1')
        self.assertEqual(lag['env']['STRATA_ADAPT_WAIT'],'1')
        self.assertNotIn('--adapt-async',lag['args'])
        async_cfg=matrix.candidate_config(base,dict(mode='kernel',frac=0,parallel=1,**{'async':True}),out,'async')
        self.assertIn('--adapt-async',async_cfg['args'])
        self.assertNotIn('STRATA_ADAPT_WAIT',async_cfg['env'])

    def test_three_means_and_incomplete_refused(self):
        rows=[]
        for size in matrix.SIZES:
            for i in range(3):
                rows.append(dict(label=str(i),input_tokens=size,prefill_tps=10+i,decode_native_tps=20+i,
                    ttft_s=1+i,prompt_ms=2+i,output_tokens=512,reused=0,quality={},output_sha256='x',
                    payload_sha256='y',native={}))
        self.assertEqual(matrix.means(rows)[0]['mean']['decode_native_tps'],21)
        self.assertEqual(matrix.means(rows)[0]['sample_stdev']['decode_native_tps'],1)
        with self.assertRaises(ValueError):matrix.means(rows[:-1])

    def test_rotation_changes_no_arithmetic_or_capacity(self):
        base=self.base();out=Path('/tmp')
        a=matrix.candidate_config(base,dict(mode='kernel',frac=0,rotate=False),out,'a')
        b=matrix.candidate_config(base,dict(mode='kernel',frac=0,rotate=True),out,'b')
        self.assertEqual(a['args'],b['args'])
        self.assertEqual(a['parallel'],b['parallel'])
        self.assertEqual(b['env']['STRATA_EXCHANGE_ROTATE'],'1')

    def test_logits_finite_shape_and_exact(self):
        from array import array
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b'
            a.write_bytes(array('f',[1,3,-2]).tobytes());b.write_bytes(a.read_bytes())
            self.assertTrue(matrix.compare_logits(a,b)['top1_equal'])
            self.assertEqual(matrix.compare_logits(a,b)['rmse'],0)
            b.write_bytes(array('f',[float('nan')]).tobytes())
            with self.assertRaises(ValueError):matrix.compare_logits(a,b)

    def test_dead_parent_owned_gpu_teardown_needs_no_api(self):
        process=SimpleNamespace(pid=123,poll=lambda:1,wait=lambda **kw:1)
        observations=iter([{456},set()])
        runner=SimpleNamespace(gpu_pids=lambda:next(observations))
        with patch.object(matrix.os,'getpgid',return_value=123),patch.object(matrix.os,'killpg') as kill:
            matrix.stop_owned(process,{},runner)
        kill.assert_called_once_with(123,signal.SIGTERM)

    def test_dead_parent_unowned_gpu_refused(self):
        process=SimpleNamespace(pid=123,poll=lambda:1)
        runner=SimpleNamespace(gpu_pids=lambda:{456})
        with patch.object(matrix.os,'getpgid',return_value=789),patch.object(matrix.os,'killpg') as kill:
            with self.assertRaises(RuntimeError):matrix.stop_owned(process,{},runner)
        kill.assert_not_called()


if __name__=='__main__':unittest.main()
