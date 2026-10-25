"""Evaluation-only filename compatibility for the frozen UMT action scorer.

The installed human_action scorer derives its target from the video basename.
Prompt-ID generation paths are retained; read-only hardlinks expose the exact
same video bytes under the original official prompt spelling for this metric.
Other dimensions, scoring source, normalization, weights and videos are unchanged.
"""
from b_common import *

CORRECTION = 'human_action_filename_contract_v1'

def input_hash_dimensions(argument):
    if argument=='all':return None
    dimensions=set(argument.split(','))
    assert dimensions and dimensions<=set(read(ROOT/'contracts/original_prompt_suite.json')['dimensions']),'Unknown requested metric dimension'
    return dimensions

def input_requires_content_hash(item,dimensions):
    # Explicit single-dimension repairs consume only that official membership.
    # Complete scoring still hashes every video; final matrix audit does too.
    return dimensions is None or bool(dimensions.intersection(item['dims']))

def action_from_path(path):
    # Exact expression in the frozen installed human_action.py, line 78.
    return str(path).split('/')[-1].lower().split('-')[0].split('person is ')[-1].split('_')[0]

def needs_action_alias(rep, dim):
    return rep != 'R0' and dim == 'human_action'

def result_folder(rep, condition, seed, scope, dim):
    base=eval_dir(rep,condition,seed,scope)
    return base/CORRECTION if needs_action_alias(rep,dim) else base

def result_paths(rep,condition,seed,scope,dim):
    base=result_folder(rep,condition,seed,scope,dim)
    return base/(dim+'_eval_results.json'),base/(dim+'_provenance.json')

def prepare_action_aliases(base, entries, records):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    metadata=[];manifest=[]
    for entry in entries:
        if 'human_action' not in entry['dimension']:continue
        prompt=entry['prompt_en'];name=prompt+'-0.mp4'
        assert Path(name).name==name and len(name.encode())<=255 and '\0' not in name
        assert prompt.startswith('A person is '),'Unexpected frozen action prompt'
        original=Path(records[prompt]['video']['path']).resolve()
        alias=base/'videos'/name;alias.parent.mkdir(parents=True,exist_ok=True)
        if not alias.exists():os.link(original,alias)
        assert not alias.is_symlink() and os.path.samefile(original,alias),'ALIAS_BYTES_NOT_IDENTICAL'
        expected=prompt.lower().split('person is ')[-1]
        assert action_from_path(alias)==expected,'ACTION_LABEL_CONTRACT_MISMATCH'
        metadata.append({**entry,'video_list':[str(alias)]})
        manifest.append(dict(prompt=prompt,prompt_id=records[prompt]['prompt_id'],original=str(original),alias=str(alias),SHA256=records[prompt]['video']['sha256'],expected_action=expected))
    assert metadata,'No official human-action entries'
    info=base/'eligible_full_info.json';mapping=base/'alias_manifest.json'
    for path,value in ((info,metadata),(mapping,manifest)):
        if path.exists():assert read(path)==value
        else:dump(path,value)
    return info,mapping,metadata

def verify_action_aliases(base):
    base=Path(base);mapping=read(base/'alias_manifest.json');entries=read(base/'eligible_full_info.json')
    assert len(mapping)==len(entries)
    bypath={row['alias']:row for row in mapping}
    for entry in entries:
        row=bypath[entry['video_list'][0]]
        assert row['prompt']==entry['prompt_en']
        assert os.path.samefile(row['original'],row['alias']) and not Path(row['alias']).is_symlink()
        assert action_from_path(row['alias'])==row['expected_action']==entry['prompt_en'].lower().split('person is ')[-1]
        assert sha(row['alias'])==row['SHA256'],'ALIAS_VIDEO_CHANGED'
    return mapping
