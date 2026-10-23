"""Fixed 128/256 source-query control; old 128 models are reused, never retrained."""
from pathlib import Path
import copy, fcntl, gc, json, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from semantic import d4rt_finetune_readiness as ready
from semantic.source_query_factorial import query_for, make_tail, evaluate
from semantic.quantized_query_mode_adapter import QuantizedQueryModeSemanticD4RT as QueryModeSemanticD4RT
from semantic.model import SemanticTail, GEOMETRY_KEYS
from semantic.ytvis import load_dataset, load_video
from semantic.run import atomic_json
from tools.load_opend4rt import load_model

ROOT = Path('semantic_artifacts/d4rt_input_resolution')
PARENT = Path('semantic_artifacts/source_query_factorial')
ARMS = {'source_head': ('source', True, 0., 0.), 'source_GT': ('source', False, 0., 0.),
        'source_SAM': ('source', False, .25, 0.), 'source_extraGT': ('source', False, 0., .25)}


def canonical_prefix(base, video, query, memory):
    with torch.no_grad():
        src = query['t_src']
        x = base.query_embedder(video=video, u=query['u'].to(video.dtype), v=query['v'].to(video.dtype),
                                t_src=src, t_tgt=src, t_cam=src)
        for block in base.decoder.blocks[:-2]:
            x = block(x, memory)
    return x


def resource_gate(manifest, dataset):
    clip = min(manifest['train'], key=lambda c: int(c['video_id']))
    assert int(clip['video_id']) == 13
    published = load_model(ready.CHECKPOINT, ready.CONFIG)
    torch.manual_seed(0)
    tail = SemanticTail(published, 26).cuda().eval()
    base = published.to('cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    query, y, _ = query_for(clip, 'train', dataset)
    query = {k: v[:, :64].cuda() for k, v in query.items()}
    video = load_video(clip, size=256)[None].cuda().to(torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        memory = base.encode_video(video)
        prefix = canonical_prefix(base, video, query, memory)
        geometry = base({'video': video, 'query': query})
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = tail(prefix.to(torch.bfloat16), memory)['semantic_logits'][0].float()
        loss = F.cross_entropy(z, y[:64].long().cuda())
    loss.backward()
    assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in tail.parameters())
    peak = torch.cuda.max_memory_allocated()
    assert peak <= 7000 * 2**20 and memory.shape[1] == 1025
    atomic_json(ROOT/'resource_gate.json', {'video_id': 13, 'shape': list(video.shape),
        'memory_tokens': memory.shape[1], 'live_prefix_dtype': str(prefix.dtype),
        'semantic_input_dtype': 'torch.bfloat16', 'GT_loss_finite': True,
        'no_optimizer_update': True, 'peak_GPU_allocated_bytes': peak,
        'budget_GPU_allocated_bytes': 7000*2**20, 'geometry_keys': sorted(geometry)})
    del published, tail, base, video, memory, prefix, geometry, z, loss
    gc.collect(); torch.cuda.empty_cache()
    print('resource gate passed', peak/2**20, 'MiB', flush=True)


def prepare(manifest, dataset):
    base = load_model(ready.CHECKPOINT, ready.CONFIG).to('cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    hashes = {}; rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for split in ('train', 'val'):
            for clip in sorted(manifest[split], key=lambda c: int(c['video_id'])):
                vid = int(clip['video_id']); q, y, xy = query_for(clip, split, dataset)
                with np.load(PARENT/'features'/split/f'{vid}.npz') as old:
                    assert np.array_equal(y.numpy(), old['labels']) and np.array_equal(xy.numpy(), old['xy'])
                    assert all(np.array_equal(v[0].numpy(), old['query_'+k]) for k,v in q.items())
                video = load_video(clip, size=256)[None].cuda().to(torch.bfloat16)
                memory = base.encode_video(video); assert memory.shape[1] == 1025
                pieces = []
                for start in range(0, len(y), 64):
                    sub = {k:v[:, start:start+64].cuda() for k,v in q.items()}
                    pieces.append(canonical_prefix(base, video, sub, memory)[0].float().cpu().numpy())
                pre = np.concatenate(pieces); path = ROOT/'features'/split/f'{vid}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, source=pre, memory=memory[0].float().cpu().numpy(),
                    labels=y.numpy().astype(np.int16), xy=xy.numpy().astype(np.int16),
                    **{'query_'+k:v[0].numpy() for k,v in q.items()})
                hashes[str(path)] = ready.sha(path)
                rows.append({'split':split, 'video_id':vid, 'queries':len(y), 'memory_tokens':1025})
                print('features256', split, vid, flush=True)
    del base; gc.collect(); torch.cuda.empty_cache()
    atomic_json(ROOT/'feature_manifest.json', {'sha256':hashes, 'rows':rows})


def load_features(split):
    data = {}
    for path in sorted((ROOT/'features'/split).glob('*.npz'), key=lambda p:int(p.stem)):
        with np.load(path) as a: r = {k:a[k].copy() for k in a.files}
        for k in ('source', 'memory'): r[k] = torch.from_numpy(r[k]).cuda().to(torch.bfloat16)
        r['y'] = torch.from_numpy(r['labels'].astype(np.int64)).cuda(); data[int(path.stem)] = r
    return data


def contrasts(results):
    pairs = [(f'{a}_256', f'{a}_128') for a in ARMS] + [
        ('source_SAM_256','source_GT_256'), ('source_SAM_256','source_extraGT_256'),
        ('source_GT_256','source_head_256')]
    out = {}
    for left,right in pairs:
        rows = []
        for seed in range(3):
            l=results[left][str(seed)]['evaluation']; r=results[right][str(seed)]['evaluation']
            d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present'] for v in r['per_video']}
            rows.append({'seed':seed, 'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],
                         'per_video_differences':d, 'video_equal_mean_difference':float(np.mean(list(d.values())))})
        out[left+'_minus_'+right]={'per_seed':rows,
            'mean_pooled_macro_difference':float(np.mean([r['pooled_macro_difference'] for r in rows])),
            'mean_video_equal_difference':float(np.mean([r['video_equal_mean_difference'] for r in rows]))}
    interaction=[]
    for seed in range(3):
        def v(a,size):return results[f'{a}_{size}'][str(seed)]['evaluation']
        videos=v('source_GT',128)['per_video']
        d={key:(v('source_SAM',256)['per_video'][key]['macro_f1_present']-v('source_GT',256)['per_video'][key]['macro_f1_present'])-(v('source_SAM',128)['per_video'][key]['macro_f1_present']-v('source_GT',128)['per_video'][key]['macro_f1_present']) for key in videos}
        pooled=(v('source_SAM',256)['pooled']['macro_f1_present']-v('source_GT',256)['pooled']['macro_f1_present'])-(v('source_SAM',128)['pooled']['macro_f1_present']-v('source_GT',128)['pooled']['macro_f1_present'])
        interaction.append({'seed':seed,'pooled_difference_in_differences':pooled,'per_video_differences_in_differences':d,'video_equal_mean_difference_in_differences':float(np.mean(list(d.values())))})
    return out, interaction


def corrected_full_interface(published,template,data,results):
 base=load_model(ready.CHECKPOINT,ready.CONFIG).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);manifest=json.loads(ready.MANIFEST.read_text());clips=sorted(manifest['val'],key=lambda c:int(c['video_id']))[:2];rows=[];batches=[]
 for clip in clips:
  vid=int(clip['video_id']);r=data[vid];q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')};batches.append({'video':load_video(clip,size=256)[None].cuda().to(torch.bfloat16),'query':q,'ids':[vid]})
 batches.append({'video':torch.cat([b['video']for b in batches]),'query':{k:torch.cat([b['query'][k]for b in batches])for k in batches[0]['query']},'ids':[int(c['video_id'])for c in clips]})
 for seed in range(3):
  for arm,(mode,head_only,_,_)in ARMS.items():
   interface=QueryModeSemanticD4RT(base,26,mode);interface.semantic_tail=make_tail(template,head_only,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));interface.eval();original_scores={}
   for j,b in enumerate(batches):
    for variant in ('original','target_camera'):
     query=dict(b['query'])
     if variant=='target_camera':query['t_tgt']=(query['t_tgt']+1)%8;query['t_cam']=(query['t_cam']+1)%8
     batch={'video':b['video'],'query':query}
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=interface(batch);ref=base(batch)
     assert all(torch.equal(out[k],ref[k])for k in GEOMETRY_KEYS);scores=out['semantic_logits'].float().cpu().numpy()
     if variant=='original':original_scores[j]=scores
     if variant=='target_camera'and mode=='source':assert np.array_equal(original_scores[j],scores)
     if j<2:
      with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as cached:
       sel=np.where(cached['video_ids']==b['ids'][0])[0][:64];expected=cached['logits'if variant=='original'else'alternate_logits'][sel];assert scores.shape[0]==1 and scores[0].shape==expected.shape;assert np.array_equal(scores[0],expected)
     path=ROOT/'interface'/f'{arm}_seed{seed}_case{j}_{variant}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**{k:v.float().cpu().numpy()for k,v in out.items()});rows.append({'arm':arm,'seed':seed,'case':j,'batch_size':len(b['video']),'video_ids':b['ids'],'variant':variant,'path':str(path),'sha256':ready.sha(path),'all_geometry_exact_base':True})
   del interface;gc.collect();torch.cuda.empty_cache()
 del base;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'interface_checks.json',{'cases':rows,'all72_geometry_exact_frozen_base':True,'all_B1_semantics_exact_cached_outputs':True,'source_target_camera_semantics_exact':True});return rows


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'protocol.json').exists() or (ROOT/'summary.json').exists():
            raise RuntimeError('Existing run: inspect, do not overwrite or retrain')
        parent_summary=json.loads((PARENT/'summary.json').read_text())
        assert json.loads((PARENT/'verification.json').read_text())['summary_sha256']==ready.sha(PARENT/'summary.json')
        pub=json.loads((PARENT/'publication.json').read_text()); assert pub['remote_contains_research_commit']
        manifest=json.loads(ready.MANIFEST.read_text()); dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'))
        schedule_path=Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json')
        schedules=json.loads(schedule_path.read_text())['schedules']
        original_inputs=json.loads((PARENT/'protocol.json').read_text())['input_sha256']
        files=[Path(p) for p in original_inputs]+[PARENT/'summary.json',PARENT/'verification.json',PARENT/'publication.json',PARENT/'recovery_amendment_v2.json',
            Path('semantic/d4rt_input_resolution.py'),Path('semantic/OVIS_D4RT_INPUT_RESOLUTION_PROTOCOL.md'),
            Path('semantic/quantized_query_mode_adapter.py'),Path('semantic/d4rt_finetune_readiness.py'),Path('tools/load_opend4rt.py')]
        files += [PARENT/f'initial_seed{s}.pt' for s in range(3)]
        files += [PARENT/f'{a}_seed{s}{suffix}' for a in ARMS for s in range(3) for suffix in ('.pt','_validation.npz')]
        files += list((PARENT/'features'/'train').glob('*.npz'))+list((PARENT/'features'/'val').glob('*.npz'))
        p={'round':106,'parent_research_commit':pub['research_commit'],'image_size':256,'clip_frames':8,
           'reference_image_size':128,'arms':{k:list(v)for k,v in ARMS.items()},'steps':500,'seeds':[0,1,2],
           'head_lr':.001,'LoRA_lr':.0001,'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))},
           'new_models':12,'reused_models':12,'semantic_tail_input_dtype':'torch.bfloat16','new_confirmation':False}
        for path,h in original_inputs.items():assert p['input_sha256'][path]==h,path
        atomic_json(ROOT/'protocol.json',p)
        resource_gate(manifest,dataset)
        prepare(manifest,dataset)
        train=load_features('train'); published=load_model(ready.CHECKPOINT,ready.CONFIG)
        results={f'{a}_{size}':{}for a in ARMS for size in (128,256)}
        for a in ARMS:
            for s in range(3):results[f'{a}_128'][str(s)]={**parent_summary['results'][a][str(s)],'reused_from_round105':True,'adapter_path':str(PARENT/f'{a}_seed{s}.pt')}
        for seed in range(3):
            torch.manual_seed(seed); template=SemanticTail(published,26).eval()
            initial=torch.load(PARENT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
            assert ready.state_sha(ready.weights(template))==ready.state_sha(initial)
            for arm,(mode,head_only,sw,ew)in ARMS.items():
                tail=make_tail(template,head_only,initial); frozen=ready.frozen_sha(tail)
                opt=torch.optim.AdamW([{'params':[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},
                                      {'params':[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)
                trace_path=ROOT/f'{arm}_seed{seed}_trace.jsonl';torch.cuda.reset_peak_memory_stats()
                with trace_path.open('x')as trace:
                    for step,row in enumerate(schedules[str(seed)]):
                        g=train[row['GT_video']];a=train[row['aux_video']]
                        gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            go=tail(g['source'][gi][None],g['memory'][None])['semantic_logits'][0].float()
                            ao=tail(a['source'][ai][None],a['memory'][None])['semantic_logits'][0].float()
                        ce=F.cross_entropy(go[:64],g['y'][gi[:64]]);aux=F.cross_entropy(ao,a['y'][ai]);extra=F.cross_entropy(go[64:],g['y'][gi[64:]])
                        loss=ce+sw*aux+ew*extra;opt.zero_grad(set_to_none=True);loss.backward()
                        assert torch.isfinite(loss)and all(q.grad is None or torch.isfinite(q.grad).all()for q in tail.parameters())
                        assert all(q.grad is None for q in tail.parameters()if not q.requires_grad)
                        if step==0:torch.save({'gradients':{n:q.grad.detach().cpu().clone()for n,q in tail.named_parameters()if q.grad is not None}},ROOT/f'{arm}_seed{seed}_step0.pt')
                        if step==499:torch.save({'adapter':ready.weights(tail),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_before_final.pt')
                        opt.step()
                        record={'step':step+1,'GT_CE':float(ce.detach()),'SAM_CE':float(aux.detach()),'extra_GT_CE':float(extra.detach()),'total':float(loss.detach())}
                        trace.write(json.dumps(record)+'\n');trace.flush()
                        if(step+1)%100==0:print('train256',seed,arm,step+1,flush=True)
                assert frozen==ready.frozen_sha(tail)
                final=ready.weights(tail);torch.save(final,ROOT/f'{arm}_seed{seed}.pt')
                results[arm+'_256'][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),
                    'adapter_path':str(ROOT/f'{arm}_seed{seed}.pt'),'trainable_parameters':sum(q.numel()for q in tail.parameters()if q.requires_grad),
                    'GT_exposures':32000,'SAM_exposures':16000 if sw else 0,'extra_GT_exposures':16000 if ew else 0,
                    'trace_path':str(trace_path),'trace_sha256':ready.sha(trace_path),'frozen_tail_unchanged':True,
                    'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated()}
                atomic_json(ROOT/'training_progress.json',results)
                del tail,opt;gc.collect();torch.cuda.empty_cache()
            del template;gc.collect();torch.cuda.empty_cache()
        selected=[Path(results[f'{a}_{size}'][str(s)]['adapter_path'])for a in ARMS for size in (128,256)for s in range(3)]
        atomic_json(ROOT/'frozen_selection.json',{'rule':'12 new final states frozen before their validation; 12 old already observed and reused',
            'adapters_sha256':{str(f):ready.sha(f)for f in selected},'protocol_sha256':ready.sha(ROOT/'protocol.json')})
        del train;gc.collect();torch.cuda.empty_cache();val=load_features('val')
        for seed in range(3):
            torch.manual_seed(seed);template=SemanticTail(published,26).eval()
            for arm,(_,ho,_,_)in ARMS.items():
                tail=make_tail(template,ho,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True))
                scores,arrays=evaluate(tail,val,'source');np.savez_compressed(ROOT/f'{arm}_seed{seed}_validation.npz',**arrays)
                results[arm+'_256'][str(seed)]['evaluation']=scores
                atomic_json(ROOT/'validation_progress.json',results)
                print('val256',seed,arm,scores['pooled']['macro_f1_present'],flush=True)
                del tail;gc.collect();torch.cuda.empty_cache()
            del template;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(0);template=SemanticTail(published,26).eval()
        corrected_full_interface(published,template,val,results)
        comps,interaction=contrasts(results)
        eligible=all(all(r['pooled_macro_difference']>0 for r in comps[k]['per_seed'])and comps[k]['mean_video_equal_difference']>0
            for k in ('source_SAM_256_minus_source_GT_256','source_SAM_256_minus_source_extraGT_256'))
        for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
        features=json.loads((ROOT/'feature_manifest.json').read_text())
        for path,h in features['sha256'].items():assert ready.sha(path)==h,path
        atomic_json(ROOT/'summary.json',{'round':106,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'contrasts':comps,
            'SAM_resolution_interaction_by_seed':interaction,'source_SAM256_eligible_for_future_external_protocol':eligible,
            'all72_interface_checks_verified_in_run':True,'geometry_main_weights_frozen':True,'geometry_truth_accuracy_evaluated':False,
            'default_changed':False,'new_confirmation':False,'feature_manifest_sha256':ready.sha(ROOT/'feature_manifest.json'),
            'interface_checks_sha256':ready.sha(ROOT/'interface_checks.json'),'resource_gate_sha256':ready.sha(ROOT/'resource_gate.json'),
            'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}|{str(f):ready.sha(f)for f in ROOT.glob('*_validation.npz')}|{str(f):ready.sha(f)for f in ROOT.glob('*_trace.jsonl')}})
        print('DONE resolution control; external eligibility',eligible,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True)
        failure=ROOT/f'failure_{time.time_ns()}.json'
        atomic_json(failure,{'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
                             'outputs_preserved':True,'automatic_retraining_or_relaxation':False})
        raise
