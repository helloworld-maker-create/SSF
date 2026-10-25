"""New-path CPU tests only: no Wan allocation, no historical B-F rerun."""
import contextlib,copy,json,unittest
from pathlib import Path
import torch
from g_accumulation import SharedExitPlan,effective_backward,EventCounters

class ToyRoot(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Linear(5,3).double();self.no_sync_calls=0;self.clip_calls=0
    def forward(self,x):return self.model(x)
    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_calls+=1;yield
    def clip_grad_norm_(self,bound):
        self.clip_calls+=1;return torch.nn.utils.clip_grad_norm_(self.parameters(),bound)

class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027);self.x=torch.randn(64,5,dtype=torch.float64);self.y=torch.randn(64,3,dtype=torch.float64)
    def test_average_and_AdamW_mapping(self):
        root=ToyRoot();ref=copy.deepcopy(root)
        opt=torch.optim.AdamW(root.parameters(),lr=2e-6,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
        ropt=torch.optim.AdamW(ref.parameters(),lr=2e-6,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
        loss=torch.nn.functional.mse_loss(ref(self.x),self.y);loss.backward()
        expected=[p.grad.clone() for p in ref.parameters()]
        errors=[]
        def before(m):
            for p,w in zip(m.parameters(),expected):
                errors.append(float((p.grad-w).abs().max()));torch.testing.assert_close(p.grad,w,rtol=1e-12,atol=1e-14)
        out=effective_backward(root,opt,lambda i:torch.nn.functional.mse_loss(root(self.x[i:i+1]),self.y[i:i+1]),64,max_norm=.1,step=True,before_clip=before)
        ref.clip_grad_norm_(.1);ropt.step()
        for p,q in zip(root.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-14)
        self.assertEqual(root.no_sync_calls,63);self.assertEqual(root.clip_calls,1);self.assertEqual(out['optimizer_steps'],1)
        self.assertEqual(len(opt.state),len(list(root.parameters())))
        print('CPU_MEAN_MAX_ABS_ERROR',max(errors),flush=True)
    def test_world_size_no_double_divide(self):
        reference=ToyRoot();torch.nn.functional.mse_loss(reference(self.x),self.y).backward()
        target=[p.grad.clone() for p in reference.parameters()]
        for W in [2,4,8]:
            rank_sums=[torch.zeros_like(g) for g in target]
            for rank in range(W):
                model=copy.deepcopy(reference);model.zero_grad(set_to_none=True)
                for micro in range(64//W):
                    idx=micro*W+rank
                    (torch.nn.functional.mse_loss(model(self.x[idx:idx+1]),self.y[idx:idx+1])/(64//W)).backward()
                for dst,p in zip(rank_sums,model.parameters()):dst.add_(p.grad/W)
            for actual,w in zip(rank_sums,target):torch.testing.assert_close(actual,w,rtol=1e-12,atol=1e-14)
    def test_events200400600(self):
        counter=EventCounters();observed=[]
        for _ in range(600):
            if counter.generator_due():counter.accepted_G()
            counter.accepted_C_and_outer()
            if counter.outer in [200,400,600]:observed.append(counter.state_dict())
        self.assertEqual([(r['outer'],r['G'],r['C'],r['EMA_updates']) for r in observed],[(200,40,200,0),(400,80,400,40),(600,120,600,80)])
    def test_shared_exit_first_item_and_restore(self):
        class P:
            same_step_across_blocks=True;last_step_only=False
            def generate_and_sync_list(self,*args):raise AssertionError('fixture must avoid RNG')
        old=P.generate_and_sync_list;fixture=[2,0,3,1,0,3,1]
        with SharedExitPlan(P,fixture) as plan:
            for micro in range(32):
                flags=P().generate_and_sync_list(7,4,'cpu')
                chosen=[i for i in range(4) if i==flags[0]]
                self.assertEqual(chosen,[2]);self.assertEqual(flags,fixture)
            self.assertEqual(plan.calls,32)
        self.assertIs(P.generate_and_sync_list,old)

if __name__=='__main__':unittest.main(verbosity=2)
