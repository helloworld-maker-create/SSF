"""Phase G control only: authorization, exact outer schedule, accounting.

No GPU work occurs on import. The immutable planning contracts stay unchanged;
real user execution grants are separate records.
"""
from g_common import *
from g_accumulation import EventCounters


def authorized(arm):
    assert arm in ['baseline','ssf']
    run=read(ROOT/'run.json')
    assert run['fingerprint']==os.environ['SSF_RUN_FINGERPRINT']
    assert run['context']['arm']==arm
    assert int(os.environ['WORLD_SIZE'])==SITE['nodes']*SITE['gpus_per_node']
    if run['context'].get('qualification_required', True):
        q=read(ROOT/'qualification/real_dmd_accumulation/complete.json')
        assert q['status']=='PASS_CLUSTER_NATIVE_DMD_B64_EMA_RESUME'
        assert q['run_fingerprint']==run['fingerprint']
        assert q['world_size']==int(os.environ['WORLD_SIZE'])
    if arm=='ssf':
        review=read(ROOT/'ssf_review.json')
        assert review['status']=='PASS' and review['ssf_authorized'] is True
    return run


def validate_boundary(counters,cursor,world):
    s=counters.state_dict();outer=s['outer']
    assert 0<=outer<=600 and s['C']==outer
    assert s['G']==(outer+4)//5
    assert s['EMA_initialized']==(outer>=200)
    assert s['EMA_updates']==max(0,s['G']-40)
    assert cursor*world==64*(s['G']+s['C'])
    return s


class Accounting:
    def __init__(self,arm,rank,attempt):
        self.arm=arm;self.rank=rank;self.attempt=attempt;self.seq=0;self.current=None
        self.categories={'training':('stage_B_training_GPUh' if arm=='baseline' else 'stage_C_training_GPUh',500),
                         'generation':('stage_B_generation_scoring_GPUh' if arm=='baseline' else 'stage_C_generation_scoring_GPUh',60)}
    def switch(self,kind):
        self.close();category,cap=self.categories[kind];budget_guard(category,cap)
        self.seq+=1;name=f'formal_{self.arm}_a{self.attempt:02d}_r{self.rank}_{self.seq:04d}'
        self.current=name
        dump(ROOT/'state'/(name+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category=category,kind=kind))
    def guard(self,kind):
        return budget_guard(*self.categories[kind])
    def close(self):
        if self.current:
            dump(ROOT/'state'/(self.current+'_finished.json'),dict(pid=os.getpid(),ended_at_epoch=time.time()))
            self.current=None


def local_order(dataset_size,rank,world,epoch):
    from torch.utils.data.distributed import DistributedSampler
    sampler=DistributedSampler(range(dataset_size),num_replicas=world,rank=rank,shuffle=True,seed=0,drop_last=True)
    sampler.set_epoch(epoch)
    return list(iter(sampler))


class DataCursor:
    def __init__(self,dataset,rank,world,cursor=0):
        self.dataset=dataset;self.rank=rank;self.world=world;self.cursor=cursor;self.orders={}
        self.length=len(dataset)//world
        assert self.length>0
    def indices(self,count):
        out=[]
        for i in range(count):
            epoch,pos=divmod(self.cursor+i,self.length)
            if epoch not in self.orders:self.orders={epoch:local_order(len(self.dataset),self.rank,self.world,epoch)}
            out.append(self.orders[epoch][pos])
        return out
    def state_dict(self):
        return dict(seed=0,shuffle=True,drop_last=True,rank=self.rank,world_size=self.world,dataset_size=len(self.dataset),accepted_rank_cursor=self.cursor,accepted_global_cursor=self.cursor*self.world,next_eight_indices=self.indices(8))
    def restore(self,s):
        assert (s['seed'],s['rank'],s['world_size'],s['dataset_size'])==(0,self.rank,self.world,len(self.dataset))
        self.cursor=s['accepted_rank_cursor']
        assert self.indices(8)==s['next_eight_indices']
