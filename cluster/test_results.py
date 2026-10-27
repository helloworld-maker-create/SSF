"""CPU/local-Git publisher tests. No GPU, training, network or remote writes."""
import argparse
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import results
from result_snapshot import collect, js, sha


def counters(outer=0,G=0):
    return dict(outer=outer,G=G,C=outer,EMA_initialized=outer>=200,EMA_updates=max(0,G-40))


def sequence(n=5,initial=None):
    c=(initial or counters()).copy();rows=[]
    while c['outer']<n:
        outer=c['outer']
        for kind in (['G','C'] if outer%5==0 else ['C']):
            if kind=='G':
                c['G']+=1
                if c['EMA_initialized']:c['EMA_updates']+=1
            else:
                c['C']+=1;c['outer']+=1
                if c['outer']==200:c['EMA_initialized']=True
            rows.append(dict(rank=0,kind=kind,outer_before=outer,counts_after=c.copy(),
                accepted_optimizer_update=True,loss_global_mean=.2 if kind=='G' else .1,
                preclip_global_norm=.3,seconds=2.,global_samples=64,local_micros=16,
                clip_calls=1,optimizer_steps=1,accepted_global_data_cursor=64*(c['G']+c['C']),
                gradient_groups_preclip=dict(Reader=0.,Writer=0.,backbone=.3),
                memory=dict(peak_allocated=1024,peak_reserved=2048)))
    return rows


class ResultTests(unittest.TestCase):
    def setUp(self):
        # Keep tiny fixtures for inspection; no deletion of user/old evidence.
        self.temp=Path(tempfile.mkdtemp(prefix='ssf_results_cpu_'))
        self.root=self.temp/'test_run';self.root.mkdir()
        self.state=self.temp/'publisher';self.state.mkdir()

    def fixture(self,n=5,attempt=0,initial=None,ranks=4):
        rows=sequence(n,initial)
        for rank in range(ranks):
            folder=self.root/'training/baseline'/('attempt_%02d'%attempt)/('rank%d'%rank)
            folder.mkdir(parents=True)
            (folder/'accepted_start.json').write_bytes(js(dict(counts=initial or counters(),
                config=dict(seed=2027,total_batch_size=64,gradient_checkpointing=False,
                            data_path='/private/location/training.txt',api_key='dummy-value-excluded'))))
            with (folder/'updates.jsonl').open('wb') as f:
                for r in rows:f.write(js({**r,'rank':rank}).replace(b'\n',b'')+b'\n')
        return rows

    def args(self):
        return argparse.Namespace(run_root=self.root,arm='baseline',world_size=4,state_dir=self.state,
            eval_root=None,remote='git@github.com:helloworld-maker-create/SSF.git',
            branch='results/test_run-baseline',interval_seconds=60)

    def test_completed_common_prefix_and_B64(self):
        self.fixture();summary,files=collect(self.root,'baseline',4)
        self.assertEqual(summary['counts'],counters(5,1));self.assertEqual(summary['accepted_updates_this_attempt'],6)
        self.assertEqual(summary['rank_observed_update_counts'],[6]*4)
        self.assertTrue(summary['rank_common_prefix_consistent'])
        self.assertNotIn('api_key',json.loads(files['training_config.json']))
        self.assertEqual(json.loads(files['training_config.json'])['data_path_basename'],'training.txt')

    def test_later_inflight_G_is_excluded(self):
        rows=self.fixture(6)
        p=self.root/'training/baseline/attempt_00/rank3/updates.jsonl'
        lines=p.read_bytes().splitlines(keepends=True);p.write_bytes(b''.join(lines[:-1]))
        summary,files=collect(self.root,'baseline',4)
        self.assertEqual(summary['counts'],counters(5,1))
        self.assertEqual(len(gzip.decompress(files['raw/updates_rank0.jsonl.gz']).splitlines()),6)

    def test_partial_last_json_line_is_not_accepted(self):
        self.fixture()
        p=self.root/'training/baseline/attempt_00/rank0/updates.jsonl'
        with p.open('ab') as f:f.write(b'{"incomplete":')
        summary,_=collect(self.root,'baseline',4)
        self.assertEqual(summary['counts']['outer'],5)

    def test_missing_rank_does_not_claim_progress(self):
        self.fixture(ranks=3)
        summary,_=collect(self.root,'baseline',4)
        self.assertEqual(summary['status'],'WAITING_FOR_RANK_LOGS')

    def test_rank_mismatch_stops_snapshot(self):
        self.fixture();p=self.root/'training/baseline/attempt_00/rank2/updates.jsonl'
        rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['loss_global_mean']=.7
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'prefix mismatch'):collect(self.root,'baseline',4)

    def test_nan_is_not_filtered_into_positive_evidence(self):
        self.fixture();p=self.root/'training/baseline/attempt_00/rank0/updates.jsonl'
        rows=[json.loads(x) for x in p.read_text().splitlines()];rows[-1]['loss_global_mean']=float('nan')
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'NONFINITE'):collect(self.root,'baseline',4)

    def test_resumed_attempt_is_not_double_counted(self):
        self.fixture(5);self.fixture(6,attempt=1,initial=counters(5,1))
        summary,_=collect(self.root,'baseline',4)
        self.assertEqual(summary['counts'],counters(6,2))
        self.assertEqual(summary['accepted_updates_this_attempt'],2)

    def test_weight_bytes_are_never_read(self):
        self.fixture();p=self.root/'training/baseline/checkpoints';p.mkdir()
        (p/'outer_000005.pt').write_bytes(b'tensor payload must never be opened')
        (p/'outer_000005.COMMITTED.json').write_bytes(js(dict(status='COMMITTED',SHA256='a'*64,bytes=42,counts=counters(5,1))))
        summary,files=collect(self.root,'baseline',4)
        self.assertEqual(len(summary['full_checkpoint_markers']),1)
        self.assertFalse(summary['checkpoint_weights_read']);self.assertTrue(all(not n.endswith('.pt') for n in files))

    def test_no_fabricated_complete_or_health_PASS(self):
        self.fixture();p=self.root/'training/baseline/TRAINING_AND_GENERATION_COMPLETE.json'
        p.write_bytes(js(dict(status='PASS')))
        with self.assertRaisesRegex(ValueError,'contradicts'):collect(self.root,'baseline',4)

    def test_secret_in_log_blocks_publication_without_echo(self):
        self.fixture();p=self.root/'logs';p.mkdir()
        sensitive='ghp_'+'A'*36
        (p/'train_baseline_rank0.log').write_text(sensitive)
        with self.assertRaisesRegex(ValueError,'CREDENTIAL_PATTERN') as caught:collect(self.root,'baseline',4)
        self.assertNotIn(sensitive,str(caught.exception))

    def test_symlink_log_is_not_followed(self):
        self.fixture();p=self.root/'logs';p.mkdir()
        (self.temp/'outside.log').write_text('outside source')
        (p/'train_baseline_rank0.log').symlink_to(self.temp/'outside.log')
        with self.assertRaisesRegex(ValueError,'Unsafe log'):collect(self.root,'baseline',4)

    def test_package_checksum_tampering_stops_before_git(self):
        self.fixture();folder,_=results.snapshot(self.args())
        (folder/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'content changed'):results.package_files(folder)

    def test_code_branch_and_credential_urls_rejected(self):
        for branch in ['main','wubin','master','results/../main','results/a/b']:
            with self.assertRaises(ValueError):results.validate_branch(branch)
        for remote in ['https://user:dummyvalue@github.com/a/b.git','https://other.example/a/b','/tmp/repo']:
            with self.assertRaises(ValueError):results.remote_name(remote)
        self.assertEqual(results.remote_name('https://github.com/helloworld-maker-create/SSF.git'),'helloworld-maker-create/SSF')

    def test_json_credential_assignment_blocks_publication(self):
        self.fixture();p=self.root/'logs';p.mkdir()
        (p/'train_baseline_rank0.log').write_text(json.dumps({'api'+'_key':'A'*32}))
        with self.assertRaisesRegex(ValueError,'CREDENTIAL_PATTERN'):collect(self.root,'baseline',4)

    def test_manifest_cannot_smuggle_weight_file(self):
        self.fixture();folder,_=results.snapshot(self.args())
        path=folder/'manifest.json';manifest=json.loads(path.read_text())
        manifest['files'].append(dict(path='weights.pt',bytes=1,sha256=sha(b'x')))
        (folder/'weights.pt').write_bytes(b'x');path.write_bytes(js(manifest))
        with self.assertRaisesRegex(ValueError,'Unapproved result artifact'):results.package_files(folder)

    def test_mock_git_failure_does_not_echo_credentials(self):
        proc=Mock(returncode=1,stdout=b'',stderr=b'private credential helper output')
        with patch('results.subprocess.run',return_value=proc) as run:
            with self.assertRaises(RuntimeError) as caught:results.git(['push','example'],self.temp)
        self.assertNotIn('private credential',str(caught.exception));self.assertEqual(run.call_count,1)

    def test_failed_upload_does_not_kill_or_retry_training(self):
        self.fixture();child=Mock();child.poll.side_effect=[None,0]
        with patch('results.publish',side_effect=RuntimeError('mock permission rejection')) as publish,\
             patch('results.time.sleep'),patch('sys.stdout',new=io.StringIO()):
            rc=results.automate(self.args(),child)
        self.assertEqual(rc,3);self.assertEqual(publish.call_count,1)
        child.kill.assert_not_called();child.terminate.assert_not_called()

    def test_cli_snapshot_needs_no_gpu_or_git(self):
        self.fixture()
        argv=['results.py','snapshot','--run-root',str(self.root),'--arm','baseline','--world-size','4']
        with patch.object(sys,'argv',argv),patch('results.git',side_effect=AssertionError),patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(results.main(),0)

    def test_run_cli_preserves_existing_command_arguments(self):
        self.fixture()
        argv=['results.py','run','--run-root',str(self.root),'--arm','baseline','--world-size','4',
              '--','existing-launcher','--run-id','test_run']
        with patch.object(sys,'argv',argv),patch('results.subprocess.Popen') as popen,\
             patch('results.automate',return_value=0):
            self.assertEqual(results.main(),0)
        self.assertEqual(popen.call_args.args[0],['existing-launcher','--run-id','test_run'])
        self.assertEqual(popen.call_args.kwargs['env']['PYTHONNOUSERSITE'],'1')

    def test_eval_reports_are_collected_not_generated(self):
        self.fixture();eroot=self.temp/'eval';report=eroot/'reports/20261027';report.mkdir(parents=True)
        (report/'summary.json').write_bytes(js(dict(status='COMPLETE',official_Total_computed=False)))
        (report/'per_video.csv').write_text('arm,score\nbaseline,0.9\n')
        summary,files=collect(self.root,'baseline',4,eval_root=eroot)
        self.assertEqual(summary['evaluation_status'],'COMPLETE')
        self.assertIn('evaluation/per_video.csv',files)

    def test_native_sequence_duplicates_are_rejected(self):
        self.fixture();p=self.root/'training/baseline/attempt_00/rank0/updates.jsonl'
        lines=p.read_bytes().splitlines(keepends=True);p.write_bytes(b''.join([lines[0],*lines]))
        with self.assertRaisesRegex(ValueError,'update order'):collect(self.root,'baseline',4)

    def test_real_local_git_two_appends_preserve_history(self):
        self.fixture();folder,_=results.snapshot(self.args())
        bare=self.temp/'remote.git'
        subprocess.run(['git','init','--quiet','--bare',str(bare)],check=True)
        remote=self.args().remote;calls=[]
        def offline(argv,cwd,stdin=None,allow_failure=False):
            calls.append(argv)
            # Replace only the transport endpoint inside this offline fixture.
            # No network/auth service is contacted and no production guard is changed.
            return results.git([str(bare) if x==remote else x for x in argv],cwd,stdin,allow_failure)
        one=results.publish(folder,remote,self.args().branch,self.state,runner=offline)
        folder2,_=results.snapshot(self.args())
        two=results.publish(folder2,remote,self.args().branch,self.state,runner=offline)
        same=results.publish(folder2,remote,self.args().branch,self.state,runner=offline)
        self.assertEqual(same['status'],'ALREADY_PUBLISHED')
        names=subprocess.check_output(['git','--git-dir',str(bare),'ls-tree','-r','--name-only',two['commit']],text=True)
        self.assertIn(one['snapshot']+'/summary.json',names);self.assertIn(two['snapshot']+'/summary.json',names)
        self.assertTrue(all('--force' not in argv and '-f' not in argv for argv in calls))
        self.assertEqual(sum(argv[0]=='push' for argv in calls),2)

    def test_concurrent_remote_advance_rejects_without_force_or_retry(self):
        self.fixture();folder,_=results.snapshot(self.args())
        bare=self.temp/'remote.git';subprocess.run(['git','init','--quiet','--bare',str(bare)],check=True)
        remote=self.args().remote
        def offline(argv,cwd,stdin=None,allow_failure=False):
            return results.git([str(bare) if x==remote else x for x in argv],cwd,stdin,allow_failure)
        first=results.publish(folder,remote,self.args().branch,self.state,runner=offline)
        folder2,_=results.snapshot(self.args());pushes=[];concurrent=[]
        def racing(argv,cwd,stdin=None,allow_failure=False):
            if argv[0]=='push':
                pushes.append(argv)
                _,tree=results.git(['--git-dir',str(bare),'rev-parse',first['commit']+'^{tree}'],self.temp)
                _,other=results.git(['--git-dir',str(bare),'commit-tree',tree.decode().strip(),'-p',first['commit'],'-m','fixture concurrent writer'],self.temp)
                concurrent.append(other.decode().strip())
                results.git(['--git-dir',str(bare),'update-ref','refs/heads/'+self.args().branch,concurrent[0]],self.temp)
            return offline(argv,cwd,stdin,allow_failure)
        with self.assertRaisesRegex(RuntimeError,'push failed'):results.publish(folder2,remote,self.args().branch,self.state,runner=racing)
        self.assertEqual(len(pushes),1);self.assertNotIn('--force',pushes[0])
        _,head=results.git(['--git-dir',str(bare),'rev-parse','refs/heads/'+self.args().branch],self.temp)
        self.assertEqual(head.decode().strip(),concurrent[0]);self.assertTrue((folder2/'manifest.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
