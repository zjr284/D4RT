"""Independent RGB/query rebuild, full training replay and matched-size audit."""
from pathlib import Path
import copy, gc, json
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail, GEOMETRY_KEYS
from semantic.ytvis import load_dataset, load_video
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha, metrics, close_dict
from tools.load_opend4rt import load_model, DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/d4rt_input_resolution')
OLD=Path('semantic_artifacts/source_query_factorial')
ARMS={'source_head':(True,0.,0.),'source_GT':(False,0.,0.),'source_SAM':(False,.25,0.),'source_extraGT':(False,0.,.25)}


def managed(n):return n.startswith('semantic_head.')or n.endswith(('.a','.b'))


def tensor_hash(q):
    import hashlib
    return hashlib.sha256(q.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def manual_prefix(base,video,q,memory):
    z=base.query_embedder(video=video,u=q['u'].to(video.dtype),v=q['v'].to(video.dtype),
                          t_src=q['t_src'],t_tgt=q['t_src'],t_cam=q['t_src'])
    for block in base.decoder.blocks[:-2]:z=block(z,memory)
    return z


def load_features(split):
    out={}
    for f in sorted((ROOT/'features'/split).glob('*.npz'),key=lambda p:int(p.stem)):
        with np.load(f)as a:r={k:a[k].copy()for k in a.files}
        r['x']=torch.from_numpy(r['source']).cuda().to(torch.bfloat16)
        r['mem']=torch.from_numpy(r['memory']).cuda().to(torch.bfloat16)
        r['y']=torch.from_numpy(r['labels'].astype(np.int64)).cuda();out[int(f.stem)]=r
    return out


def tail_for(template,state,ho):
    tail=copy.deepcopy(template).cuda().eval()
    with torch.no_grad():
        for n,q in tail.named_parameters():
            if n in state:q.copy_(state[n].to(q.device))
            if ho and n.endswith(('.a','.b')):q.requires_grad_(False)
    return tail


def optimizer(tail):
    body=[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')]
    head=[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')]
    return torch.optim.AdamW([{'params':body,'lr':.0001},{'params':head,'lr':.001}],weight_decay=.01)


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    assert s['round']==106 and p['steps']==500 and p['image_size']==256 and p['clip_frames']==8
    assert p['new_models']==12 and p['reused_models']==12 and s['protocol_sha256']==sha(ROOT/'protocol.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    old=json.loads((OLD/'summary.json').read_text());ov=json.loads((OLD/'verification.json').read_text())
    assert ov['summary_sha256']==sha(OLD/'summary.json')and json.loads((OLD/'publication.json').read_text())['remote_contains_research_commit']
    for path,h in s['outputs_sha256'].items():assert sha(path)==h,path
    feature=json.loads((ROOT/'feature_manifest.json').read_text());assert s['feature_manifest_sha256']==sha(ROOT/'feature_manifest.json')and len(feature['sha256'])==80
    for path,h in feature['sha256'].items():assert sha(path)==h,path
    selected=json.loads((ROOT/'frozen_selection.json').read_text());required={str(root/f'{a}_seed{k}.pt')for root in (ROOT,OLD)for a in ARMS for k in range(3)}
    assert set(selected['adapters_sha256'])==required
    for path,h in selected['adapters_sha256'].items():assert sha(path)==h,path
    gate=json.loads((ROOT/'resource_gate.json').read_text());assert s['resource_gate_sha256']==sha(ROOT/'resource_gate.json')
    assert gate['video_id']==13 and gate['shape']==[1,8,3,256,256]and gate['memory_tokens']==1025
    assert gate['no_optimizer_update']and gate['GT_loss_finite']and gate['peak_GPU_allocated_bytes']<=7000*2**20
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for split in ('train','val'):
            for c in sorted(manifest[split],key=lambda c:int(c['video_id'])):
                vid=int(c['video_id'])
                if split=='train':
                    with np.load(Path('semantic_artifacts/teacher_ovis')/f'{vid}.npz')as a:maps=a['labels'].copy()
                else:maps=np.stack([dataset.gt_label_map(c,t)for t in range(8)])
                rng=np.random.default_rng(42+vid);coords=[]
                for t in range(8):
                    for pool in (np.argwhere(maps[t]>0),np.argwhere(maps[t]==0)):
                        if len(pool):coords.extend((int(u),int(v),t)for v,u in pool[rng.choice(len(pool),128,replace=len(pool)<128)])
                xy=np.array(coords);n=len(xy);tgt=rng.integers(0,8,n);cam=rng.integers(0,8,n)
                y=maps[xy[:,2],xy[:,1],xy[:,0]];u=(xy[:,0]/127).astype(np.float32);v=(xy[:,1]/127).astype(np.float32)
                with np.load(ROOT/'features'/split/f'{vid}.npz')as a:stored={k:a[k].copy()for k in a.files}
                with np.load(OLD/'features'/split/f'{vid}.npz')as a:
                    assert np.array_equal(stored['xy'],a['xy'])and np.array_equal(stored['labels'],a['labels'])
                    for k in ('u','v','t_src','t_tgt','t_cam'):assert np.array_equal(stored['query_'+k],a['query_'+k])
                assert np.array_equal(xy,stored['xy'])and np.array_equal(y,stored['labels'])
                for k,values in [('u',u),('v',v),('t_src',xy[:,2]),('t_tgt',tgt),('t_cam',cam)]:assert np.array_equal(values,stored['query_'+k])
                video=load_video(c,size=256)[None].cuda().to(torch.bfloat16);memory=base.encode_video(video)
                assert memory.shape[1]==1025 and np.array_equal(memory[0].float().cpu().numpy(),stored['memory'])
                for start in range(0,n,64):
                    q={k:torch.from_numpy(stored['query_'+k][start:start+64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')}
                    z=manual_prefix(base,video,q,memory)[0].float().cpu().numpy()
                    assert np.array_equal(z,stored['source'][start:start+64])
            print('Verified all RGB/query/source-prefix256',split,flush=True)
    del base;gc.collect();torch.cuda.empty_cache()
    published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml')
    schedules=json.loads(Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json').read_text())['schedules']
    train=load_features('train');replays={}
    for seed in range(3):
        torch.manual_seed(seed);template=SemanticTail(published,26).eval()
        initial=torch.load(OLD/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
        for arm,(ho,sw,ew)in ARMS.items():
            record=s['results'][arm+'_256'][str(seed)];tail=tail_for(template,initial,ho);opt=optimizer(tail)
            assert {n:tensor_hash(q)for n,q in initial.items()}==record['initial_adapter_sha256']
            assert sum(q.numel()for q in tail.parameters()if q.requires_grad)==record['trainable_parameters']==(33306 if ho else 217626)
            frozen={n:tensor_hash(q)for n,q in tail.named_parameters()if not managed(n)}
            trace=[json.loads(line)for line in Path(record['trace_path']).read_text().splitlines()]
            assert sha(record['trace_path'])==record['trace_sha256']and len(trace)==len(schedules[str(seed)])==500
            before=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True)
            final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)
            max_trace_error=0.;gradient_error=0.;before_error=0.
            for index,row in enumerate(schedules[str(seed)]):
                gt=train[row['GT_video']];aux=train[row['aux_video']]
                gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    gz=tail(gt['x'][gi][None],gt['mem'][None])['semantic_logits'][0].float()
                    az=tail(aux['x'][ai][None],aux['mem'][None])['semantic_logits'][0].float()
                ce=F.cross_entropy(gz[:64],gt['y'][gi[:64]])
                ae=F.cross_entropy(az,aux['y'][ai]);ee=F.cross_entropy(gz[64:],gt['y'][gi[64:]])
                loss=ce+sw*ae+ew*ee
                expected={'GT_CE':float(ce.detach()),'SAM_CE':float(ae.detach()),'extra_GT_CE':float(ee.detach()),'total':float(loss.detach())}
                assert trace[index]['step']==index+1
                for k,value in expected.items():
                    error=abs(trace[index][k]-value);max_trace_error=max(max_trace_error,error)
                    assert np.isclose(trace[index][k],value,rtol=1e-6,atol=1e-7),(seed,arm,index,k,error)
                opt.zero_grad(set_to_none=True);loss.backward()
                if index==0:
                    saved=torch.load(ROOT/f'{arm}_seed{seed}_step0.pt',map_location='cpu',weights_only=True)['gradients']
                    gradients={n:q.grad for n,q in tail.named_parameters()if q.grad is not None};assert set(saved)==set(gradients)
                    for n,g in gradients.items():
                        assert managed(n);torch.testing.assert_close(g.cpu(),saved[n],rtol=5e-4,atol=5e-6)
                        gradient_error=max(gradient_error,float((g.cpu()-saved[n]).abs().max()))
                if index==499:
                    for n,q in tail.named_parameters():
                        if n in before['adapter']:
                            torch.testing.assert_close(q.detach().cpu(),before['adapter'][n],rtol=3e-4,atol=2e-6)
                            before_error=max(before_error,float((q.detach().cpu()-before['adapter'][n]).abs().max()))
                    current=opt.state_dict();assert current['param_groups']==before['optimizer']['param_groups']
                    assert set(current['state'])==set(before['optimizer']['state'])
                    for key,state in before['optimizer']['state'].items():
                        assert float(state['step'])==499
                        for name,tensor in state.items():assert torch.equal(current['state'][key][name].cpu(),tensor.cpu())
                opt.step()
            parameter_error=0.
            for n,q in tail.named_parameters():
                if n in final:
                    torch.testing.assert_close(q.detach().cpu(),final[n],rtol=3e-4,atol=2e-6)
                    parameter_error=max(parameter_error,float((q.detach().cpu()-final[n]).abs().max()))
            assert {n:tensor_hash(q)for n,q in final.items()}==record['final_adapter_sha256']
            assert frozen=={n:tensor_hash(q)for n,q in tail.named_parameters()if not managed(n)}
            if ho:assert all(torch.equal(final[n],initial[n])for n in final if n.endswith(('.a','.b')))
            assert record['GT_exposures']==32000 and record['SAM_exposures']==(16000 if sw else 0)and record['extra_GT_exposures']==(16000 if ew else 0)
            replays[f'{arm}_{seed}']={'steps_replayed':500,'trace_max_abs_error':max_trace_error,'first_gradient_max_abs_error':gradient_error,
                                    'step499_parameter_max_abs_error':before_error,'step499_optimizer_states_exact':True,'final_parameter_max_abs_error':parameter_error}
            print('Verified complete500 trajectory',seed,arm,flush=True)
            del tail,opt;gc.collect();torch.cuda.empty_cache()
        del template;gc.collect();torch.cuda.empty_cache()
    del train;gc.collect();torch.cuda.empty_cache()
    val=load_features('val');rebuilt={};confusions={}
    for seed in range(3):
        torch.manual_seed(seed);template=SemanticTail(published,26).eval()
        for arm,(ho,_,_)in ARMS.items():
            for size,root in ((128,OLD),(256,ROOT)):
                key=f'{arm}_{size}';saved=np.load(root/f'{arm}_seed{seed}_validation.npz');per={}
                if size==256:
                    tail=tail_for(template,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True),ho);offset=0
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for vid,r in val.items():
                            z=np.concatenate([tail(r['x'][j:j+64][None],r['mem'][None])['semantic_logits'][0].float().cpu().numpy()for j in range(0,len(r['labels']),64)])
                            n=len(z);assert np.array_equal(z,saved['logits'][offset:offset+n])and np.array_equal(z,saved['alternate_logits'][offset:offset+n])
                            assert np.array_equal(r['labels'],saved['labels'][offset:offset+n])and np.all(saved['video_ids'][offset:offset+n]==vid);offset+=n
                    assert offset==40960;del tail;gc.collect();torch.cuda.empty_cache()
                else:
                    logged=s['results'][key][str(seed)];assert logged['reused_from_round105']
                    for name,value in old['results'][arm][str(seed)].items():assert logged[name]==value
                for vid in sorted(set(saved['video_ids'])):
                    idx=saved['video_ids']==vid;per[str(int(vid))]=metrics(saved['labels'][idx],saved['logits'][idx].argmax(1))[0]
                pooled,cm=metrics(saved['labels'],saved['logits'].argmax(1));record=s['results'][key][str(seed)]['evaluation']
                close_dict(pooled,record['pooled']);close_dict(per,record['per_video']);assert record['target_camera_variant_agreement']==1.
                assert np.array_equal(saved['logits'],saved['alternate_logits'])
                rebuilt.setdefault(key,{})[str(seed)]={'pooled':pooled,'per_video':per};confusions[f'{key}_{seed}']=cm.tolist();saved.close()
        del template;gc.collect();torch.cuda.empty_cache();print('Verified four new and four reused validation outputs seed',seed,flush=True)
    # End-to-end reference is assembled directly from frozen base and tail.
    base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    clips=sorted(manifest['val'],key=lambda c:int(c['video_id']))[:2];batches=[]
    for c in clips:
        r=val[int(c['video_id'])];q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')}
        batches.append({'video':load_video(c,size=256)[None].cuda().to(torch.bfloat16),'query':q,'ids':[int(c['video_id'])]})
    batches.append({'video':torch.cat([b['video']for b in batches]),'query':{k:torch.cat([b['query'][k]for b in batches])for k in batches[0]['query']},'ids':[int(c['video_id'])for c in clips]})
    manual={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j,b in enumerate(batches):
            for variant in ('original','target_camera'):
                q=dict(b['query'])
                if variant=='target_camera':q['t_tgt']=(q['t_tgt']+1)%8;q['t_cam']=(q['t_cam']+1)%8
                mem=base.encode_video(b['video']);pre=manual_prefix(base,b['video'],q,mem)
                manual[(j,variant)]={'geometry':base({'video':b['video'],'query':q}),'prefix':pre.to(torch.bfloat16),'memory':mem}
    checks=json.loads((ROOT/'interface_checks.json').read_text())
    assert s['interface_checks_sha256']==sha(ROOT/'interface_checks.json')and len(checks['cases'])==72
    torch.manual_seed(0);template=SemanticTail(published,26).eval()
    for seed in range(3):
        for arm,(ho,_,_)in ARMS.items():
            tail=tail_for(template,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True),ho);original={}
            records=[r for r in checks['cases']if r['arm']==arm and r['seed']==seed]
            assert len(records)==6 and len({(r['case'],r['variant'])for r in records})==6
            for r in records:
                j=r['case'];variant=r['variant'];ref=manual[(j,variant)]
                assert r['video_ids']==batches[j]['ids']and r['batch_size']==len(batches[j]['video'])and sha(r['path'])==r['sha256']
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=tail(ref['prefix'],ref['memory'])['semantic_logits'].float().cpu().numpy()
                with np.load(r['path'])as saved:
                    assert np.array_equal(z,saved['semantic_logits'])
                    for k in GEOMETRY_KEYS:assert np.array_equal(ref['geometry'][k].float().cpu().numpy(),saved[k])
                if variant=='original':original[j]=z
                else:assert np.array_equal(original[j],z)
                if j<2:
                    with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as saved:
                        selected=np.flatnonzero(saved['video_ids']==batches[j]['ids'][0])[:64]
                        assert np.array_equal(z[0],saved['logits'][selected])
            del tail;gc.collect();torch.cuda.empty_cache()
        print('Verified24 same-input256 geometry/semantics cases seed',seed,flush=True)
    pairs=[(f'{a}_256',f'{a}_128')for a in ARMS]+[('source_SAM_256','source_GT_256'),('source_SAM_256','source_extraGT_256'),('source_GT_256','source_head_256')]
    for left,right in pairs:
        rows=[]
        for seed in range(3):
            l=rebuilt[left][str(seed)];r=rebuilt[right][str(seed)]
            d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']}
            rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],
                'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
        stored=s['contrasts'][left+'_minus_'+right];assert stored['per_seed']==rows
        assert abs(stored['mean_pooled_macro_difference']-np.mean([r['pooled_macro_difference']for r in rows]))<1e-12
        assert abs(stored['mean_video_equal_difference']-np.mean([r['video_equal_mean_difference']for r in rows]))<1e-12
    for seed,stored in enumerate(s['SAM_resolution_interaction_by_seed']):
        def v(a,size):return rebuilt[f'{a}_{size}'][str(seed)]
        d={key:(v('source_SAM',256)['per_video'][key]['macro_f1_present']-v('source_GT',256)['per_video'][key]['macro_f1_present'])-(v('source_SAM',128)['per_video'][key]['macro_f1_present']-v('source_GT',128)['per_video'][key]['macro_f1_present'])for key in v('source_GT',128)['per_video']}
        pooled=(v('source_SAM',256)['pooled']['macro_f1_present']-v('source_GT',256)['pooled']['macro_f1_present'])-(v('source_SAM',128)['pooled']['macro_f1_present']-v('source_GT',128)['pooled']['macro_f1_present'])
        assert stored['seed']==seed and stored['per_video_differences_in_differences']==d
        assert pooled==stored['pooled_difference_in_differences']and stored['video_equal_mean_difference_in_differences']==float(np.mean(list(d.values())))
    eligible=all(all(r['pooled_macro_difference']>0 for r in s['contrasts'][k]['per_seed'])and s['contrasts'][k]['mean_video_equal_difference']>0 for k in ('source_SAM_256_minus_source_GT_256','source_SAM_256_minus_source_extraGT_256'))
    assert eligible==s['source_SAM256_eligible_for_future_external_protocol']
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    for path,h in feature['sha256'].items():assert sha(path)==h,path
    atomic_json(ROOT/'verification.json',{'round':106,'summary_sha256':sha(ROOT/'summary.json'),
        'verifier_sha256':sha('semantic/verify_d4rt_input_resolution.py'),'all80_RGB_queries_and256_prefixes_rebuilt_exact':True,
        'all12_full500_step_training_replays':replays,'all12_new_validation_logits_exact':True,
        'all12_old_validation_metrics_and_reuse_verified':True,'all24_confusions':confusions,
        'all72_manual_same_input_geometry_semantic_and_B1_cache_checks_exact':True,
        'all_contrasts_interaction_and_eligibility_verified':True,'all_original_input_fingerprints_unchanged':True,
        'no_geometry_truth_accuracy_claim':True,'new_confirmation':False})
    print('DONE independent input-resolution verification',flush=True)

if __name__=='__main__':main()
