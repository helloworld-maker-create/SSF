"""CPU checks for launch boundaries; no model or CUDA imports."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch
from topology import monitor_slots


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cfg = launch.read(launch.REPO / 'cluster/site.example.json')
        self.cfg.update(transport='local', python=sys.executable, repo=str(launch.REPO),
                        output=str(self.folder / 'output'), hosts=['localhost'])

    def site(self, cfg=None):
        path = self.folder / 'site.json'
        path.write_text(json.dumps(cfg or self.cfg))
        return path

    def test_b64_topologies_and_invalid_worlds(self):
        for nodes, local in [(1, 2), (1, 4), (1, 8), (2, 8), (4, 8), (8, 8)]:
            cfg = {**self.cfg, 'transport': 'ssh', 'nodes': nodes, 'gpus_per_node': local}
            self.assertEqual(64 % (launch.config(self.site(cfg))['nodes'] * local), 0)
        for nodes, local in [(32, 8), (3, 8), (1, 0)]:
            with self.assertRaises(ValueError):
                launch.config(self.site({**self.cfg, 'nodes': nodes, 'gpus_per_node': local}))

    def test_monitor_collectives_balanced_without_duplicate_records(self):
        for count in [16, 96]:
            rows = list(range(count))
            for world in [2, 4, 8, 16, 32, 64]:
                calls = [list(monitor_slots(rows, world, rank)) for rank in range(world)]
                self.assertEqual(len({len(batch) for batch in calls}), 1)
                actual = [row for batch in calls for active, row in batch if active]
                self.assertEqual(sorted(actual), rows)

    def test_inventory_alias_validation_and_dedup(self):
        inventory = self.folder / 'hosts'
        inventory.write_text('# nodes\nnode-a|10.1.1.1\nnode-b|10.1.1.2\n')
        cfg = {**self.cfg, 'transport': 'ssh', 'hosts': [], 'inventory': str(inventory)}
        self.assertEqual(launch.hosts(cfg), ['10.1.1.1', '10.1.1.2'])
        for hosts in [['10.1.1.1', '10.1.1.1'], ['-oProxyCommand=bad'], ['host;touch /tmp/bad']]:
            with self.assertRaises(ValueError):
                launch.hosts({**cfg, 'hosts': hosts})

    def test_ssh_arguments_roundtrip_as_data(self):
        cfg = {**self.cfg, 'transport': 'ssh', 'ssh_user': 'example', 'ssh_identity': None}
        argv = ['/path with space/python', '-c', 'print("$(do_not_execute)")', 'line1\nline2']
        remote = launch.remote(cfg, '10.1.1.1', argv)
        self.assertEqual(shlex.split(remote[-1]), ['exec', *argv])

    def test_unknown_settings_and_environment_rejected(self):
        for extra in [dict(total_batch_size=8), dict(environment={'PYTHONPATH': '/elsewhere'}), dict(ssh_user='-oProxyCommand=x')]:
            with self.assertRaises(ValueError):
                launch.config(self.site({**self.cfg, **extra}))

    def test_dry_run_no_node_or_model_work(self):
        result = subprocess.run([sys.executable, '-B', str(launch.REPO / 'cluster/launch.py'),
                                 '--config', str(self.site()), '--dry-run'],
                                text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value['status'], 'PLAN_ONLY')
        self.assertFalse(value['node_contacted'])
        self.assertFalse(value['training_started'])
        self.assertFalse((self.folder / 'output').exists())

    def test_missing_config_and_bad_init_do_not_launch(self):
        with patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('unexpected GPU launch')):
            with patch.object(sys, 'argv', ['launch', '--config', str(self.folder / 'missing.json')]):
                with self.assertRaisesRegex(ValueError, 'Site config absent'):
                    launch.main()
        path = self.folder / 'existing.json'
        path.write_text('preserve')
        with self.assertRaises(FileExistsError):
            launch.init_config(path)
        self.assertEqual(path.read_text(), 'preserve')

    def test_source_manifest_excludes_local_receipts(self):
        # The receipt created after setup must not change the source signature.
        expected = launch.code_manifest()
        receipts = launch.REPO / 'cluster/run_receipts'
        receipts.mkdir(exist_ok=True)
        path = receipts / 'cpu_test.local.json'
        try:
            path.write_text('{"private":"local"}')
            self.assertEqual(launch.code_manifest(), expected)
        finally:
            path.unlink()
        self.assertNotIn('cluster/site.local.json', expected)

    def test_failing_node_does_not_start_training(self):
        cfg = {**self.cfg, 'transport': 'ssh', 'hosts': ['10.1.1.1']}
        with patch.object(launch, 'capture', side_effect=ValueError('disk full')):
            with patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('unexpected launch')):
                with self.assertRaisesRegex(ValueError, 'no GPU workers launched'):
                    launch.select_nodes(cfg)

    def test_no_model_imported_by_controller(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; sys.path.insert(0,sys.argv[1]); import launch; '
            'assert not any(n=="torch" or n.startswith(("model.","wan.")) for n in sys.modules)',
            str(launch.REPO / 'cluster')], check=True)
        self.assertEqual(result.returncode, 0)

    def pipeline(self, result_status='PASS_CLUSTER_NATIVE_DMD_B64_EMA_RESUME', shared=True, hours=1):
        """Exercise controller decisions with a fake transport, never fake DMD."""
        cluster = self.folder / 'controller/cluster'
        cluster.mkdir(parents=True)
        (cluster / 'site.example.json').write_text(json.dumps(self.cfg))
        site = self.site()
        report = {'hostname': 'node-one', 'gpus': []}
        handshake = {}
        calls = []

        def action(cfg, host, kind, payload):
            run = payload['run']
            if kind == 'handshake':
                handshake['token'] = payload['token']
                return {'path': run['root'] + '/token.json'}
            if kind == 'read' and payload['path'].endswith('/token.json'):
                if not shared:
                    raise FileNotFoundError('shared output token missing')
                return dict(handshake)
            if kind == 'read' and payload['path'].endswith('/complete.json'):
                return dict(status=result_status, run_fingerprint=run['fingerprint'], projected_training_hours=hours)
            if kind == 'prepare_train':
                return {'attempt': 0}
            if kind == 'read' and payload['path'].endswith('/TRAINING_AND_GENERATION_COMPLETE.json'):
                return {'status': 'PASS_600_OUTER_120G_600C_EMA80'}
            return {'ready': True}

        def group(cfg, run, mode, arm, resume=None):
            calls.append(mode)

        patches = [patch.object(launch, 'REPO', self.folder / 'controller'),
                   patch.object(launch, 'check_code'), patch.object(launch, 'code_manifest', return_value={}),
                   patch.object(launch, 'select_nodes', return_value=[('localhost', report)]),
                   patch.object(launch, 'deploy'), patch.object(launch, 'remote_action', side_effect=action),
                   patch.object(launch, 'run_group', side_effect=group),
                   patch.object(sys, 'argv', ['launch', '--config', str(site), '--with-qualification'])]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            try:
                launch.main()
            except (ValueError, FileNotFoundError) as exc:
                return calls, str(exc)
        return calls, None

    def test_controller_qualifies_before_training(self):
        calls, error = self.pipeline()
        self.assertIsNone(error)
        self.assertEqual(calls, ['qualify', 'train'])

    def test_failed_qualification_never_trains(self):
        calls, error = self.pipeline(result_status='FAIL_NATIVE_DMD')
        self.assertEqual(calls, ['qualify'])
        self.assertIn('did not pass', error)

    def test_unshared_output_never_starts_gpu_work(self):
        calls, error = self.pipeline(shared=False)
        self.assertEqual(calls, [])
        self.assertIn('shared output', error)

    def test_measured_wall_cost_gate(self):
        calls, error = self.pipeline(hours=100)
        self.assertEqual(calls, ['qualify'])
        self.assertIn('exceeds wall_hours', error)

    def test_supervisor_returns_failure_and_terminates_owned_group(self):
        script = str(launch.REPO / 'cluster/supervise.py')
        result = subprocess.run([sys.executable, script, sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(result.returncode, 7)
        alive = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        marker = self.folder / 'started'
        child = subprocess.Popen([sys.executable, script, sys.executable, '-c',
                                  'from pathlib import Path; import sys,time; Path(sys.argv[1]).write_text("ready"); time.sleep(20)', str(marker)])
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(marker.exists())
            child.terminate()
            child.wait(timeout=5)
            self.assertIsNone(alive.poll(), 'supervisor must not affect unrelated processes')
        finally:
            for proc in [child, alive]:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
