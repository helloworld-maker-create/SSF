"""Focused algebra/step-boundary regression for the candidate communication path."""
import copy, unittest, torch
from test_accumulation_cpu import ToyRoot
from g_accumulation_sync_candidate import effective_backward


class CandidateTests(unittest.TestCase):
    def test_mean_clip_and_single_AdamW_update(self):
        torch.manual_seed(2027)
        x=torch.randn(64,5,dtype=torch.float64);y=torch.randn(64,3,dtype=torch.float64)
        model=ToyRoot();reference=copy.deepcopy(model)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-6,betas=(0.,.999),eps=1e-8,weight_decay=.01)
        ropt=torch.optim.AdamW(reference.parameters(),lr=2e-6,betas=(0.,.999),eps=1e-8,weight_decay=.01)
        torch.nn.functional.mse_loss(reference(x),y).backward()
        expected=[p.grad.clone() for p in reference.parameters()]
        fixed=[p.detach().clone() for p in model.parameters()]
        def loss_fn(i):
            for p,q in zip(model.parameters(),fixed):self.assertTrue(torch.equal(p,q))
            self.assertFalse(opt.state)
            return torch.nn.functional.mse_loss(model(x[i:i+1]),y[i:i+1])
        def before_clip(root):
            for p,g in zip(root.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=1e-12,atol=1e-14)
        out=effective_backward(model,opt,loss_fn,64,max_norm=.1,step=True,before_clip=before_clip)
        reference.clip_grad_norm_(.1);ropt.step()
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-14)
        self.assertEqual(model.no_sync_calls,0);self.assertEqual(model.clip_calls,1)
        self.assertEqual(out['optimizer_steps'],1)
        self.assertTrue(all(float(s['step'])==1 for s in opt.state.values()))


if __name__=='__main__':unittest.main(verbosity=2)
