"""Independent exact deterministic trajectories, BG features and full outputs."""
from pathlib import Path
import copy,gc,json,os
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics,close_dict
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/sam_background_matched');OLD=Path('semantic_artifacts/d4rt_input_resolution')
PREP=Path('semantic_artifacts/sam_confident_background');INITIAL=Path('semantic_artifacts/source_query_factorial')
COEFF={'GT':(0.,0.,0.,0.),'SAM_FG':(.25,0.,0.,0.),'SAM_FG_double':(.5,0.,0.,0.),'SAM_FGBG':(.25,.25,0.,0.),'GTextra_FGBG':(0.,0.,.25,.25)}

def managed(n):return n.startswith('semantic_head.')or n.endswith(('.a','.b'))

def hash_tensor(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def prefix(base,video,q,memory):
    src=q['t_src'];x=base.query_embedder(video=video,u=q['u'].to(video.dtype),v=q['v'].to(video.dtype),t_src=src,t_tgt=src,t_cam=src)
    for block in base.decoder.blocks[:-2]:x=block(x,memory)
    return x

def data(split):
    out={}
    for path in sorted((OLD/'features'/split).glob('*.npz'),key=lambda p:int(p.stem)):
        with np.load(path)as a:r={k:a[k].copy()for k in a.files}
        r['x']=torch.from_numpy(r['source']).cuda().to(torch.bfloat16);r['mem']=torch.from_numpy(r['memory']).cuda().to(torch.bfloat16)
        r['y']=torch.from_numpy(r['labels'].astype(np.int64)).cuda()
        if split=='train':
            with np.load(ROOT/'background_features'/path.name)as a:
                r['bx']=torch.from_numpy(a['source'].copy()).cuda().to(torch.bfloat16);r['bxy']=a['xy'].copy();assert np.all(a['labels']==0)
        out[int(path.stem)]=r
    return out

def tail_for(template,state):
    m=copy.deepcopy(template).cuda().eval()
    with torch.no_grad():
        for n,q in m.named_parameters():
            if n in state:q.copy_(state[n].to(q.device))
    return m

def optimizer(m):
    return torch.optim.AdamW([{'params':[q for n,q in m.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},
                             {'params':[q for n,q in m.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)

def temporal(z,y):
    pred=z.argmax(1);fg=y>0;bg=y==0
    return {'queries':len(y),'foreground_correct_fraction':float(np.count_nonzero(pred[fg]==y[fg])/np.count_nonzero(fg)),
            'background_false_positive_fraction':float(np.count_nonzero(pred[bg])/np.count_nonzero(bg)),
            'predicted_foreground_fraction':float(np.count_nonzero(pred)/len(y))}

def main():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8';torch.use_deterministic_algorithms(True)
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    assert p['round']==109 and p['steps']==500 and p['coefficients']=={k:list(v)for k,v in COEFF.items()}
    assert p['deterministic_algorithms']and p['CUBLAS_WORKSPACE_CONFIG']==':4096:8'and s['protocol_sha256']==sha(ROOT/'protocol.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    for path,h in s['outputs_sha256'].items():assert sha(path)==h,path
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());features=json.loads((ROOT/'feature_manifest.json').read_text())
    assert s['feature_manifest_sha256']==sha(ROOT/'feature_manifest.json')and len(features['sha256'])==60
    for path,h in features['sha256'].items():assert sha(path)==h,path
    expanded=json.loads((ROOT/'expanded_schedules.json').read_text());parent=json.loads((PREP/'paired_schedules.json').read_text())
    assert s['expanded_schedules_sha256']==sha(ROOT/'expanded_schedules.json')and expanded['parent_schedules_sha256']==sha(PREP/'paired_schedules.json')
    required={str(ROOT/f'{a}_seed{k}.pt')for a in COEFF for k in range(3)};selected=json.loads((ROOT/'frozen_selection.json').read_text())
    assert set(selected['adapters_sha256'])==required
    for path,h in selected['adapters_sha256'].items():assert sha(path)==h,path
    points={int(c['video_id']):set()for c in manifest['train']}
    for seed,rows in expanded['schedules'].items():
        assert len(rows)==500
        for got,source in zip(rows,parent['schedules'][seed]):
            for k,v in source.items():assert got[k]==v
            points[source['aux_video']].update(tuple(v)for v in source['SAM_BG_xy'])
    base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    unique_count=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for clip in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
            vid=int(clip['video_id']);xy=np.array(sorted(points[vid]),dtype=np.int64);unique_count+=len(xy)
            with np.load(ROOT/'background_features'/f'{vid}.npz')as a:bg={k:a[k].copy()for k in a.files}
            assert np.array_equal(bg['xy'],xy)and np.all(bg['labels']==0)
            with np.load(PREP/'teachers'/f'{vid}.npz')as a:assert np.all(a['labels'][xy[:,2],xy[:,1],xy[:,0]]==0)
            video=load_video(clip,size=256)[None].cuda().to(torch.bfloat16);mem=base.encode_video(video)
            with np.load(OLD/'features'/'train'/f'{vid}.npz')as a:assert np.array_equal(mem[0].float().cpu().numpy(),a['memory'])
            for start in range(0,len(xy),64):
                sub=xy[start:start+64];q={'u':torch.from_numpy((sub[:,0]/127).astype(np.float32)).cuda()[None],
                    'v':torch.from_numpy((sub[:,1]/127).astype(np.float32)).cuda()[None],'t_src':torch.from_numpy(sub[:,2].copy()).cuda()[None]}
                z=prefix(base,video,q,mem)[0].float().cpu().numpy();assert np.array_equal(z,bg['source'][start:start+64])
    assert unique_count==features['unique_background_points'];del base;gc.collect();torch.cuda.empty_cache()
    print('Verified60 BG feature packages, original RGB/memory and paired identities',flush=True)
    published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');train=data('train');replays={}
    for seed in range(3):
        torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(INITIAL/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
        for arm,c in COEFF.items():
            logged=s['results'][arm][str(seed)];tail=tail_for(template,initial);opt=optimizer(tail)
            assert logged['initial_adapter_sha256']=={n:hash_tensor(q)for n,q in initial.items()}
            assert logged['trainable_parameters']==sum(q.numel()for q in tail.parameters()if q.requires_grad)==217626
            frozen={n:hash_tensor(q)for n,q in tail.named_parameters()if not managed(n)}
            trace=[json.loads(line)for line in Path(logged['trace_path']).read_text().splitlines()]
            assert len(trace)==500 and sha(logged['trace_path'])==logged['trace_sha256']
            assert logged['deterministic_first8_mirror_checks']==[{'step':j+1,'all_managed_parameters_and_losses_exact':True}for j in range(8)]
            saved_grad=torch.load(ROOT/f'{arm}_seed{seed}_step0.pt',map_location='cpu',weights_only=True)['gradients']
            before=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True)
            final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)
            for step,row in enumerate(expanded['schedules'][str(seed)]):
                g=train[row['GT_video']];a=train[row['aux_video']]
                gi=torch.tensor(row['GT_indices']+row['extra_GT_indices']+row['extra_GT_BG_indices'],device='cuda')
                ai=torch.tensor(row['SAM_indices'],device='cuda');bi=torch.tensor(row['SAM_BG_indices'],device='cuda')
                assert np.array_equal(a['bxy'][row['SAM_BG_indices']],row['SAM_BG_xy'])and np.array_equal(a['xy'][row['SAM_indices'],2],a['bxy'][row['SAM_BG_indices'],2])
                assert torch.all(g['y'][gi[96:]]==0)and torch.all(a['y'][ai]>0)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    gz=tail(g['x'][gi][None],g['mem'][None])['semantic_logits'][0].float()
                    az=tail(torch.cat([a['x'][ai],a['bx'][bi]])[None],a['mem'][None])['semantic_logits'][0].float()
                gt=F.cross_entropy(gz[:64],g['y'][gi[:64]]);fg=F.cross_entropy(az[:32],a['y'][ai]);bg=F.cross_entropy(az[32:],torch.zeros(32,dtype=torch.long,device='cuda'))
                egf=F.cross_entropy(gz[64:96],g['y'][gi[64:96]]);egb=F.cross_entropy(gz[96:],g['y'][gi[96:]])
                loss=gt+c[0]*fg+c[1]*bg+c[2]*egf+c[3]*egb
                expected={'step':step+1,'GT_CE':float(gt.detach()),'SAM_FG_CE':float(fg.detach()),'SAM_BG_CE':float(bg.detach()),
                          'extra_GT_FG_CE':float(egf.detach()),'extra_GT_BG_CE':float(egb.detach()),'total':float(loss.detach())}
                assert expected==trace[step],(seed,arm,step,expected,trace[step])
                opt.zero_grad(set_to_none=True);loss.backward()
                if step==0:
                    grads={n:q.grad for n,q in tail.named_parameters()if q.grad is not None};assert set(grads)==set(saved_grad)
                    for n,q in grads.items():assert managed(n)and torch.equal(q.cpu(),saved_grad[n]),(seed,arm,n)
                if step==499:
                    for n,q in tail.named_parameters():
                        if n in initial:assert torch.equal(q.detach().cpu(),before['adapter'][n]),(seed,arm,n)
                    state=opt.state_dict();assert state['param_groups']==before['optimizer']['param_groups']
                    for k,values in before['optimizer']['state'].items():
                        assert float(values['step'])==499
                        for n,v in values.items():assert torch.equal(state['state'][k][n].cpu(),v.cpu())
                opt.step()
                if(step+1)%100==0:
                    checkpoint=torch.load(ROOT/f'{arm}_seed{seed}_resume_step{step+1}.pt',map_location='cpu',weights_only=True);assert checkpoint['step']==step+1
                    for n,q in tail.named_parameters():
                        if n in initial:assert torch.equal(q.detach().cpu(),checkpoint['adapter'][n])
            for n,q in tail.named_parameters():
                if n in initial:assert torch.equal(q.detach().cpu(),final[n]),(seed,arm,n)
            assert logged['final_adapter_sha256']=={n:hash_tensor(q)for n,q in final.items()}
            assert frozen=={n:hash_tensor(q)for n,q in tail.named_parameters()if not managed(n)}
            for k,v in [('GT_exposures',32000),('SAM_FG_exposures',16000 if c[0]else 0),('SAM_BG_exposures',16000 if c[1]else 0),('extra_GT_FG_exposures',16000 if c[2]else 0),('extra_GT_BG_exposures',16000 if c[3]else 0)]:assert logged[k]==v
            replays[f'{arm}_{seed}']={'all500_step_losses_exact':True,'first_gradients_exact':True,'step499_adapter_and_optimizer_exact':True,'all5_resume_checkpoints_exact':True,'final_parameters_exact':True}
            print('Verified exact full500 deterministic trajectory',seed,arm,flush=True);del tail,opt;gc.collect();torch.cuda.empty_cache()
        del template;gc.collect();torch.cuda.empty_cache()
    del train;gc.collect();torch.cuda.empty_cache();val=data('val');rebuilt={a:{}for a in COEFF};confusions={}
    times=np.concatenate([r['xy'][:,2]for r in val.values()])
    for seed in range(3):
        torch.manual_seed(seed);template=SemanticTail(published,26).eval()
        for arm in COEFF:
            tail=tail_for(template,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));saved=np.load(ROOT/f'{arm}_seed{seed}_validation.npz');offset=0;per={}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for vid,r in val.items():
                    z=np.concatenate([tail(r['x'][j:j+64][None],r['mem'][None])['semantic_logits'][0].float().cpu().numpy()for j in range(0,len(r['labels']),64)])
                    n=len(z);assert np.array_equal(z,saved['logits'][offset:offset+n])and np.array_equal(z,saved['alternate_logits'][offset:offset+n])
                    assert np.array_equal(r['labels'],saved['labels'][offset:offset+n])and np.all(saved['video_ids'][offset:offset+n]==vid)
                    per[str(vid)]=metrics(r['labels'],z.argmax(1))[0];offset+=n
            assert offset==40960
            pooled,cm=metrics(saved['labels'],saved['logits'].argmax(1));logged=s['results'][arm][str(seed)]
            close_dict(pooled,logged['evaluation']['pooled']);close_dict(per,logged['evaluation']['per_video']);assert logged['evaluation']['target_camera_variant_agreement']==1.
            for name,mask in [('source',times==0),('later',times>0)]:close_dict(temporal(saved['logits'][mask],saved['labels'][mask]),logged['temporal_populations'][name])
            rebuilt[arm][str(seed)]={'pooled':pooled,'per_video':per};confusions[f'{arm}_{seed}']=cm.tolist();saved.close();del tail;gc.collect();torch.cuda.empty_cache()
        del template;gc.collect();torch.cuda.empty_cache();print('Verified five complete validation outputs seed',seed,flush=True)
    base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    clips=sorted(manifest['val'],key=lambda c:int(c['video_id']))[:2];batches=[]
    for clip in clips:
        r=val[int(clip['video_id'])];q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')}
        batches.append({'video':load_video(clip,size=256)[None].cuda().to(torch.bfloat16),'query':q,'ids':[int(clip['video_id'])]})
    batches.append({'video':torch.cat([b['video']for b in batches]),'query':{k:torch.cat([b['query'][k]for b in batches])for k in batches[0]['query']},'ids':[int(c['video_id'])for c in clips]})
    manual={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j,b in enumerate(batches):
            for variant in ('original','target_camera'):
                q=dict(b['query'])
                if variant=='target_camera':q['t_tgt']=(q['t_tgt']+1)%8;q['t_cam']=(q['t_cam']+1)%8
                mem=base.encode_video(b['video'])
                manual[(j,variant)]={'geometry':base({'video':b['video'],'query':q}),'prefix':prefix(base,b['video'],q,mem).to(torch.bfloat16),'memory':mem}
    checks=json.loads((ROOT/'interface_checks.json').read_text())
    assert s['interface_checks_sha256']==sha(ROOT/'interface_checks.json')and len(checks['cases'])==90
    torch.manual_seed(0);template=SemanticTail(published,26).eval()
    for seed in range(3):
        for arm in COEFF:
            tail=tail_for(template,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));original={}
            records=[r for r in checks['cases']if r['seed']==seed and r['arm']==arm]
            assert len(records)==6 and len({(r['case'],r['variant'])for r in records})==6
            for record in records:
                j=record['case'];variant=record['variant'];reference=manual[(j,variant)]
                assert record['video_ids']==batches[j]['ids']and record['batch_size']==len(batches[j]['video'])and sha(record['path'])==record['sha256']
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=tail(reference['prefix'],reference['memory'])['semantic_logits'].float().cpu().numpy()
                with np.load(record['path'])as cached:
                    assert np.array_equal(z,cached['semantic_logits'])
                    for k in GEOMETRY_KEYS:assert np.array_equal(reference['geometry'][k].float().cpu().numpy(),cached[k])
                if variant=='original':original[j]=z
                else:assert np.array_equal(original[j],z)
                if j<2:
                    with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as cached:
                        idx=np.flatnonzero(cached['video_ids']==batches[j]['ids'][0])[:64];assert np.array_equal(z[0],cached['logits'][idx])
            del tail;gc.collect();torch.cuda.empty_cache()
        print('Verified30 exact geometry and canonical semantic interfaces seed',seed,flush=True)
    rebuilt_contrasts={}
    for right in ('SAM_FG','SAM_FG_double','GTextra_FGBG','GT'):
        rows=[]
        for seed in range(3):
            left=rebuilt['SAM_FGBG'][str(seed)];r=rebuilt[right][str(seed)]
            d={v:left['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']}
            rows.append({'seed':seed,'pooled_macro_difference':left['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],
                         'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
        key='SAM_FGBG_minus_'+right;stored=s['contrasts'][key]
        assert stored['per_seed']==rows
        assert stored['mean_pooled_macro_difference']==float(np.mean([r['pooled_macro_difference']for r in rows]))
        assert stored['mean_video_equal_difference']==float(np.mean([r['video_equal_mean_difference']for r in rows]))
        rebuilt_contrasts[key]=stored
    eligible=all(all(r['pooled_macro_difference']>0 for r in c['per_seed'])and c['mean_video_equal_difference']>0 for c in rebuilt_contrasts.values())
    assert eligible==s['SAM_FGBG_eligible_for_future_external_protocol']
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    for path,h in features['sha256'].items():assert sha(path)==h,path
    atomic_json(ROOT/'verification.json',{'round':109,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_sam_background_matched.py'),
        'all60_BG_features_original_RGB_memory_and_paired_identities_exact':True,'all15_full500_deterministic_training_replays':replays,
        'all15_complete_validation_logits_confusions_and_temporal_populations_exact':True,'all15_confusions':confusions,
        'all90_manual_geometry_semantics_and_B1_cache_checks_exact':True,'all_four_contrasts_and_eligibility_verified':True,
        'all_original_inputs_and_models_unchanged':True,'no_new_confirmation_or_geometry_truth_accuracy_claim':True})
    print('DONE independent exact matched SAM background verification',flush=True)
if __name__=='__main__':main()
