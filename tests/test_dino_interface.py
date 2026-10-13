import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from semantic.dino_interface import DinoSemanticD4RT
from semantic.model import GEOMETRY_KEYS
from semantic.oracle_probe import file_hash


class FakeGeometry(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.))
    def forward(self, batch):
        q = batch['query']; value = (q['u'] + q['v'] + q['t_tgt'] + 2*q['t_cam'])[..., None] * self.weight
        return {**{k: value.clone() for k in GEOMETRY_KEYS}, 'extra_geometry_field': value.square()}


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.))
    def forward_features(self, image):
        patches = F.avg_pool2d(image, 14).flatten(2).transpose(1, 2).repeat(1, 1, 128)*self.weight
        return {'x_norm_patchtokens': patches}


def main():
    model = DinoSemanticD4RT(FakeGeometry(), FakeDino()).eval()
    with torch.no_grad():
        model.semantic_head.classifier.weight.zero_(); model.semantic_head.classifier.bias.zero_()
        model.semantic_head.classifier.weight[1, 0] = 1
    video = torch.zeros(2, 2, 3, 128, 128)
    video[:, 0, 0] = .2; video[:, 1, 0] = .8; video[1, :, 1] = .9
    ramp = torch.linspace(0, 1, 128)[None, None, :].expand(2, 128, 128)
    video[:, 0, 0] = ramp
    query = {'u': torch.tensor([[0., .5, 1.], [.1, .8, .9]]), 'v': torch.tensor([[0., .75, 1.], [.2, .7, .9]]),
             't_src': torch.tensor([[0, 0, 1], [1, 0, 1]]), 't_tgt': torch.zeros(2, 3, dtype=torch.long), 't_cam': torch.ones(2, 3, dtype=torch.long)}
    saved = {k: v.clone() for k, v in query.items()}; rgb = video.clone(); batch = {'video': video, 'query': query}
    expected = model.base(batch); result = model(batch)
    for k in expected: assert torch.equal(result[k], expected[k])
    assert result['semantic_logits'].shape == (2, 3, 26) and result['semantic_labels'].shape == (2, 3)
    assert torch.equal(result['semantic_logits'][..., 1:], result['semantic_raw_logits'][..., 1:])
    assert torch.equal(result['semantic_logits'][..., 0], result['semantic_raw_logits'][..., 0] - 1.75)
    assert torch.equal(result['semantic_labels'], result['semantic_logits'].argmax(-1))
    changed = {k: v.clone() for k, v in query.items()}; changed['t_tgt'] = 1-query['t_tgt']; changed['t_cam'] = 1-query['t_cam']
    other = model({'video': video, 'query': changed})
    assert torch.equal(result['semantic_logits'], other['semantic_logits'])
    assert not torch.equal(result['xyz_3d'], other['xyz_3d'])
    single = {k: v[:, :1] for k, v in query.items()}
    assert model({'video': video, 'query': single})['semantic_logits'].shape == (2, 1, 26)
    source_changed = {k: v.clone() for k, v in query.items()}; source_changed['t_src'] = 1-query['t_src']
    assert not torch.equal(model.semantic_features(video, query), model.semantic_features(video, source_changed))
    assert not torch.equal(result['semantic_raw_logits'][0, 0], result['semantic_raw_logits'][0, 1])
    assert torch.equal(video, rgb) and all(torch.equal(query[k], saved[k]) for k in query)
    paired = {'video': video.to(torch.bfloat16), 'semantic_video': video, 'query': query}
    paired_result = model(paired)
    for k, value in model.base(paired).items(): assert torch.equal(paired_result[k], value)
    assert torch.equal(paired_result['semantic_logits'], result['semantic_logits'])
    assert torch.equal(paired['video'], video.to(torch.bfloat16)) and torch.equal(video, rgb)
    altered_rgb = video.clone(); altered_rgb[0, 0, 0, 0, 0] = .95
    for semantic_rgb in (altered_rgb, video.to(torch.bfloat16), video[:, :1], video.to('meta')):
        try: model({**paired, 'semantic_video': semantic_rgb})
        except ValueError: pass
        else: raise AssertionError('Invalid paired semantic RGB accepted')
    assert all(not p.requires_grad for p in model.parameters())
    try: model.train()
    except ValueError: pass
    else: raise AssertionError('Frozen adapter must reject training')
    bad_queries = []
    for field, value in [('u', -0.01), ('v', 1.01), ('u', float('nan')), ('t_src', -1), ('t_cam', 2)]:
        q = {k: v.clone() for k, v in query.items()}; q[field][0, 0] = value; bad_queries.append(q)
    q = {k: v.clone() for k, v in query.items()}; q['t_src'] = q['t_src'].float(); bad_queries.append(q)
    q = {k: v.clone() for k, v in query.items()}; q['u'] = []; bad_queries.append(q)
    for q in bad_queries:
        try: model({'video': video, 'query': q})
        except ValueError: pass
        else: raise AssertionError('Invalid query accepted')
    for v in (video[:, :, :, :64], video.long(), video+2, video*float('nan')):
        try: model({'video': v, 'query': query})
        except ValueError: pass
        else: raise AssertionError('Invalid RGB video accepted')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'head.pt'; torch.save({'model': model.semantic_head.state_dict()}, path)
        fresh = DinoSemanticD4RT(FakeGeometry(), FakeDino()).eval(); fresh.load_head(path, file_hash(path))
        assert torch.equal(model(batch)['semantic_logits'], fresh(batch)['semantic_logits'])
        try: fresh.load_head(path, '0'*64)
        except ValueError: pass
        else: raise AssertionError('Changed checkpoint accepted')
        torch.save({'model': {'classifier.weight': torch.zeros(41, 384), 'classifier.bias': torch.zeros(41)}}, path)
        try: fresh.load_head(path)
        except ValueError: pass
        else: raise AssertionError('Wrong class count accepted')
    print('Geometry delegation, B2/Q1/Q3, time invariance, source/coordinate dependence, calibration, bounds and reload passed')


if __name__ == '__main__': main()
