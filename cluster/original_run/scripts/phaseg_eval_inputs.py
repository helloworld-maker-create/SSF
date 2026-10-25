"""CPU-only validation and units for the frozen Phase G evaluation rows."""
from g_common import *


def selected_rows(arm, outer, weight, seed):
    assert arm in ['baseline', 'ssf'] and outer in [200, 400, 600]
    assert weight in ['raw', 'EMA'] and seed in [2026, 2027]
    stage = 'B' if arm == 'baseline' else 'C'
    rows = [r for r in matrix(stage) if r['arm'] == arm
            and int(r['completed_outer']) == outer and r['weight'] == weight
            and int(r['seed']) == seed and r['condition'] == 'normal']
    assert len(rows) == (48 if outer == 600 else 8)
    assert len({r['prompt'] for r in rows}) == len(rows)
    return rows


def load_generation_records(rows):
    records = {}
    for row in rows:
        p = ROOT/'generation/records'/(row['video_id']+'.json')
        r = read(p)
        assert r['matrix_row'] == row and r['latent_finite']
        v = r['video']; target = ROOT/row['expected_output']
        assert v['path'] == str(target) and v['lossless']
        assert (v['frames'], v['fps'], v['width'], v['height']) == (81, 16, 832, 480)
        assert target.is_file() and sha(target) == v['SHA256']
        assert r['weight'] == row['weight']
        assert r['State_writer_calls'] == (7 if row['arm'] == 'ssf' else 0)
        records[row['prompt']] = {**r, 'video': {**v, 'sha256': v['SHA256']}}
    assert len({r['checkpoint_SHA256'] for r in records.values()}) == 1
    return records


def video_dimension_raw(dimension, native_video_score):
    # Installed VBench MUSIQ emits per-video 0..100 but aggregate 0..1.
    # This adapter is for analysis only, never rewriting native metric output.
    score = float(native_video_score)
    return score/100 if dimension == 'imaging_quality' else score

