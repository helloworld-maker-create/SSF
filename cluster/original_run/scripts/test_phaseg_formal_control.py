"""CPU integration checks for the formal outer/EMA/data control path."""
import unittest
from phaseg_formal_control import EventCounters,validate_boundary,DataCursor,local_order,authorized
from g_common import matrix


class FormalTests(unittest.TestCase):
    def test_full600_native_schedule_ends_after_all_critic_updates(self):
        c=EventCounters();cursor=0;events=[];points=[];ema_exists=False;refreshes=0
        for outer in range(600):
            self.assertEqual(c.outer,outer)
            for kind in (['G','C'] if c.generator_due() else ['C']):
                cursor+=32;events.append((outer,kind))
                if kind=='G':
                    c.accepted_G()
                    if ema_exists:refreshes+=1
                else:
                    c.accepted_C_and_outer()
                    if c.EMA_initialized and not ema_exists:ema_exists=True
            validate_boundary(c,cursor,2)
            if c.outer in [200,400,600]:points.append((c.outer,c.G,c.C,c.EMA_updates,refreshes,cursor*2))
        self.assertEqual(points,[(200,40,200,0,0,15360),(400,80,400,40,40,30720),(600,120,600,80,80,46080)])
        last_g=max(i for i,e in enumerate(events) if e[1]=='G')
        self.assertEqual(events[last_g:],[(595,'G'),(595,'C'),(596,'C'),(597,'C'),(598,'C'),(599,'C')])
    def test_sampler_cursor_roundtrip_and_epoch_crossing(self):
        dataset=list(range(130))
        for rank in [0,1]:
            c=DataCursor(dataset,rank,2);c.cursor=60
            expected=local_order(130,rank,2,0)[60:]+local_order(130,rank,2,1)[:11]
            self.assertEqual(c.indices(16),expected)
            state=c.state_dict();d=DataCursor(dataset,rank,2);d.restore(state)
            self.assertEqual(c.indices(32),d.indices(32))
            self.assertEqual(state['accepted_global_cursor'],120)
    def test_baseline_grant_does_not_authorize_SSF(self):
        self.assertEqual(authorized('baseline')['stage_B_execution'],'AUTHORIZED')
        with self.assertRaises((FileNotFoundError,AssertionError)):authorized('ssf')
    def test_partial_outer_or_wrong_cursor_rejected(self):
        c=EventCounters();c.accepted_G()
        with self.assertRaises(AssertionError):validate_boundary(c,32,2)
        c.accepted_C_and_outer();validate_boundary(c,64,2)
        with self.assertRaises(AssertionError):validate_boundary(c,63,2)
    def test_frozen_monitor_matrix_has_balanced_rank_assignments(self):
        rows=matrix('B')
        for outer,expected in [(200,32),(400,32),(600,192)]:
            selected=[r for r in rows if r['arm']=='baseline' and int(r['completed_outer'])==outer and r['condition']=='normal']
            self.assertEqual(len(selected),expected)
            for weight in ['raw','EMA']:
                subset=[r for r in selected if r['weight']==weight]
                self.assertEqual(len(subset),expected//2)
                self.assertEqual(len(subset[0::2]),len(subset[1::2]))


if __name__=='__main__':unittest.main(verbosity=2)
