"""Deterministic matched foreground/background supervision beside frozen geometry."""
from pathlib import Path
import copy,fcntl,gc,json,os,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from semantic import d4rt_finetune_readiness as ready
from semantic import d4rt_input_resolution as resolution
from semantic.source_query_factorial import make_tail,evaluate
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.quantized_query_mode_adapter import QuantizedQueryModeSemanticD4RT as QueryModeSemanticD4RT
from semantic.ytvis import load_video
from semantic.run import atomic_json
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/sam_background_matched')
PARENT=Path('semantic_artifacts/sam_confident_background')
INITIAL=Path('semantic_artifacts/source_query_factorial')
COEFFICIENTS={'GT':(0.,0.,0.,0.),'SAM_FG':(.25,0.,0.,0.),'SAM_FG_double':(.5,0.,0.,0.),
              'SAM_FGBG':(.25,.25,0.,0.),'GTextra_FGBG':(0.,0.,.25,.25)}
ARMS={key:('source',False,0.,0.)for key in COEFFICIENTS}


def prepare_background(manifest,schedules):
    all_coords={int(c['video_id']):[]for c in manifest['train']}
    for rows in schedules.values():
        for row in rows:all_coords[row['aux_video']].extend(row['SAM_BG_xy'])
    unique={vid:np.unique(np.asarray(values,dtype=np.int64),axis=0)for vid,values in all_coords.items()}
    base=load_model(ready.CHECKPOINT,ready.CONFIG).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    hashes={};mapping={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
            vid=int(c['video_id']);xy=unique[vid]
            assert len(xy)>0
            with np.load(PARENT/'teachers'/f'{vid}.npz')as a:assert np.all(a['labels'][xy[:,2],xy[:,1],xy[:,0]]==0)
            with np.load(resolution.ROOT/'features'/'train'/f'{vid}.npz')as a:memory=torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None]
            video=load_video(c,size=256)[None].cuda().to(torch.bfloat16);assert torch.equal(base.encode_video(video),memory)
            pre=[]
            for j in range(0,len(xy),64):
                part=xy[j:j+64];src=torch.from_numpy(part[:,2].copy()).cuda()[None]
                u=torch.from_numpy((part[:,0]/127).astype(np.float32)).cuda()[None];v=torch.from_numpy((part[:,1]/127).astype(np.float32)).cuda()[None]
                z=base.query_embedder(video=video,u=u.to(video.dtype),v=v.to(video.dtype),t_src=src,t_tgt=src,t_cam=src)
                for block in base.decoder.blocks[:-2]:z=block(z,memory)
                pre.append(z[0].float().cpu().numpy())
            path=ROOT/'background_features'/f'{vid}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,source=np.concatenate(pre),xy=xy.astype(np.int16),labels=np.zeros(len(xy),dtype=np.int16))
            hashes[str(path)]=ready.sha(path);mapping[vid]={tuple(map(int,p)):i for i,p in enumerate(xy)}
            print('BG features',vid,len(xy),flush=True)
    del base;gc.collect();torch.cuda.empty_cache()
    expanded={}
    for seed,rows in schedules.items():expanded[seed]=[{**row,'SAM_BG_indices':[mapping[row['aux_video']][tuple(xy)]for xy in row['SAM_BG_xy']]}for row in rows]
    atomic_json(ROOT/'feature_manifest.json',{'sha256':hashes,'unique_background_points':sum(len(v)for v in unique.values())})
    atomic_json(ROOT/'expanded_schedules.json',{'schedules':expanded,'parent_schedules_sha256':ready.sha(PARENT/'paired_schedules.json')})
    return expanded


def train_data():
    data=resolution.load_features('train')
    for vid,r in data.items():
        with np.load(ROOT/'background_features'/f'{vid}.npz')as a:
            r['BG_source']=torch.from_numpy(a['source'].copy()).cuda().to(torch.bfloat16)
            r['BG_labels']=torch.from_numpy(a['labels'].astype(np.int64)).cuda()
    return data


def optimizer(tail):
    return torch.optim.AdamW([{'params':[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},
                             {'params':[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)


def loss_terms(tail,row,data,coeff):
    g=data[row['GT_video']];a=data[row['aux_video']]
    gi=torch.tensor(row['GT_indices']+row['extra_GT_indices']+row['extra_GT_BG_indices'],device='cuda')
    ai=torch.tensor(row['SAM_indices'],device='cuda');bi=torch.tensor(row['SAM_BG_indices'],device='cuda')
    assert torch.all(g['y'][gi[96:]]==0)and torch.all(a['BG_labels'][bi]==0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        gz=tail(g['source'][gi][None],g['memory'][None])['semantic_logits'][0].float()
        az=tail(torch.cat([a['source'][ai],a['BG_source'][bi]])[None],a['memory'][None])['semantic_logits'][0].float()
    terms={'GT_CE':F.cross_entropy(gz[:64],g['y'][gi[:64]]),
           'SAM_FG_CE':F.cross_entropy(az[:32],a['y'][ai]),'SAM_BG_CE':F.cross_entropy(az[32:],a['BG_labels'][bi]),
           'extra_GT_FG_CE':F.cross_entropy(gz[64:96],g['y'][gi[64:96]]),'extra_GT_BG_CE':F.cross_entropy(gz[96:],g['y'][gi[96:]])}
    total=terms['GT_CE']+coeff[0]*terms['SAM_FG_CE']+coeff[1]*terms['SAM_BG_CE']+coeff[2]*terms['extra_GT_FG_CE']+coeff[3]*terms['extra_GT_BG_CE']
    return total,terms


def describe_population(z,y):
    pred=z.argmax(1);fg=y>0;bg=y==0
    return {'queries':len(y),'foreground_correct_fraction':float(np.mean(pred[fg]==y[fg])),
            'background_false_positive_fraction':float(np.mean(pred[bg]>0)),'predicted_foreground_fraction':float(np.mean(pred>0))}


def comparisons(results):
    out={}
    for right in ('SAM_FG','SAM_FG_double','GTextra_FGBG','GT'):
        rows=[]
        for seed in range(3):
            l=results['SAM_FGBG'][str(seed)]['evaluation'];r=results[right][str(seed)]['evaluation']
            d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']}
            rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],
                         'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
        out['SAM_FGBG_minus_'+right]={'per_seed':rows,'mean_pooled_macro_difference':float(np.mean([r['pooled_macro_difference']for r in rows])),
            'mean_video_equal_difference':float(np.mean([r['video_equal_mean_difference']for r in rows]))}
    return out


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
 del base;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'interface_checks.json',{'cases':rows,'all90_geometry_exact_frozen_base':True,'all_B1_semantics_exact_cached_outputs':True,'source_target_camera_semantics_exact':True});return rows



def main():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8';torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'protocol.json').exists()or(ROOT/'summary.json').exists():raise RuntimeError('Existing run: inspect before any explicit recovery')
        parent=json.loads((PARENT/'summary.json').read_text());assert parent['eligible_for_separate_matched_training_protocol']
        assert json.loads((PARENT/'verification.json').read_text())['summary_sha256']==ready.sha(PARENT/'summary.json')
        pub=json.loads((PARENT/'publication.json').read_text());assert pub['remote_contains_research_commit']
        manifest=json.loads(ready.MANIFEST.read_text());schedules=json.loads((PARENT/'paired_schedules.json').read_text())['schedules']
        files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,ready.ANN,PARENT/'summary.json',PARENT/'verification.json',PARENT/'publication.json',PARENT/'paired_schedules.json',
            Path('semantic/sam_background_matched.py'),Path('semantic/OVIS_SAM_BACKGROUND_MATCHED_PROTOCOL.md'),Path('semantic/model.py'),Path('semantic/source_query_factorial.py'),
            Path('semantic/d4rt_input_resolution.py'),Path('semantic/quantized_query_mode_adapter.py'),Path('semantic/query_mode_adapter.py'),Path('semantic/source_query_adapter.py'),Path('tools/load_opend4rt.py')]
        files +=[Path(r['path'])for r in parent['teachers']]+[INITIAL/f'initial_seed{s}.pt'for s in range(3)]
        files +=list((resolution.ROOT/'features'/'train').glob('*.npz'))+list((resolution.ROOT/'features'/'val').glob('*.npz'))
        files +=[Path(f)for split in ('train','val')for c in manifest[split]for f in c['frames']]
        p={'round':109,'parent_research_commit':pub['research_commit'],'coefficients':{k:list(v)for k,v in COEFFICIENTS.items()},'steps':500,'seeds':[0,1,2],
           'head_lr':.001,'LoRA_lr':.0001,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
           'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'torch':torch.__version__,'CUDA':torch.version.cuda,
           'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))},'new_confirmation':False}
        atomic_json(ROOT/'protocol.json',p);expanded=prepare_background(manifest,schedules);data=train_data()
        published=load_model(ready.CHECKPOINT,ready.CONFIG);results={a:{}for a in ARMS}
        for seed in range(3):
            torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(INITIAL/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
            assert ready.state_sha(initial)==ready.state_sha(ready.weights(template))
            for arm,coeff in COEFFICIENTS.items():
                tail=make_tail(template,False,initial);opt=optimizer(tail);frozen=ready.frozen_sha(tail)
                mirror=make_tail(template,False,initial);mirror_opt=optimizer(mirror);mirror_checks=[]
                trace_path=ROOT/f'{arm}_seed{seed}_trace.jsonl';torch.cuda.reset_peak_memory_stats()
                with trace_path.open('x')as trace:
                    for j,row in enumerate(expanded[str(seed)]):
                        loss,terms=loss_terms(tail,row,data,coeff);opt.zero_grad(set_to_none=True);loss.backward()
                        assert torch.isfinite(loss)and all(q.grad is None or torch.isfinite(q.grad).all()for q in tail.parameters())
                        assert all(q.grad is None for q in tail.parameters()if not q.requires_grad)
                        if j==0:torch.save({'gradients':{n:q.grad.detach().cpu().clone()for n,q in tail.named_parameters()if q.grad is not None}},ROOT/f'{arm}_seed{seed}_step0.pt')
                        if j==499:torch.save({'adapter':ready.weights(tail),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_before_final.pt')
                        opt.step()
                        if j<8:
                            m_loss,m_terms=loss_terms(mirror,row,data,coeff)
                            assert torch.equal(m_loss.detach(),loss.detach())and all(torch.equal(m_terms[k].detach(),v.detach())for k,v in terms.items())
                            mirror_opt.zero_grad(set_to_none=True);m_loss.backward();mirror_opt.step()
                            assert all(torch.equal(q.detach(),dict(mirror.named_parameters())[n].detach())for n,q in tail.named_parameters()if n in initial)
                            mirror_checks.append({'step':j+1,'all_managed_parameters_and_losses_exact':True})
                            if j==7:del mirror,mirror_opt;gc.collect();torch.cuda.empty_cache()
                        record={'step':j+1,**{k:float(v.detach())for k,v in terms.items()},'total':float(loss.detach())}
                        trace.write(json.dumps(record)+'\n');trace.flush()
                        if(j+1)%100==0:
                            torch.save({'step':j+1,'adapter':ready.weights(tail),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_resume_step{j+1}.pt')
                            atomic_json(ROOT/'training_progress.json',{'seed':seed,'arm':arm,'step':j+1,'trace_path':str(trace_path),'loss':record,'completed':results})
                            print('trainFGBG',seed,arm,j+1,flush=True)
                assert frozen==ready.frozen_sha(tail)
                final=ready.weights(tail);torch.save(final,ROOT/f'{arm}_seed{seed}.pt')
                results[arm][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),
                    'trainable_parameters':sum(q.numel()for q in tail.parameters()if q.requires_grad),'GT_exposures':32000,
                    'SAM_FG_exposures':16000 if coeff[0]else 0,'SAM_BG_exposures':16000 if coeff[1]else 0,
                    'extra_GT_FG_exposures':16000 if coeff[2]else 0,'extra_GT_BG_exposures':16000 if coeff[3]else 0,
                    'trace_path':str(trace_path),'trace_sha256':ready.sha(trace_path),'deterministic_first8_mirror_checks':mirror_checks,
                    'frozen_tail_unchanged':True,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated()}
                atomic_json(ROOT/'completed_training.json',results);del tail,opt;gc.collect();torch.cuda.empty_cache()
            del template;gc.collect();torch.cuda.empty_cache()
        selected=[ROOT/f'{a}_seed{s}.pt'for a in ARMS for s in range(3)]
        atomic_json(ROOT/'frozen_selection.json',{'rule':'all15 new final states frozen before old validation scoring',
            'adapters_sha256':{str(f):ready.sha(f)for f in selected},'protocol_sha256':ready.sha(ROOT/'protocol.json')})
        del data;gc.collect();torch.cuda.empty_cache();val=resolution.load_features('val')
        val_t=np.concatenate([r['xy'][:,2]for r in val.values()])
        for seed in range(3):
            torch.manual_seed(seed);template=SemanticTail(published,26).eval()
            for arm in ARMS:
                tail=make_tail(template,False,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True))
                scores,arrays=evaluate(tail,val,'source');np.savez_compressed(ROOT/f'{arm}_seed{seed}_validation.npz',**arrays)
                results[arm][str(seed)]['evaluation']=scores
                results[arm][str(seed)]['temporal_populations']={name:describe_population(arrays['logits'][mask],arrays['labels'][mask])for name,mask in [('source',val_t==0),('later',val_t>0)]}
                atomic_json(ROOT/'validation_progress.json',results);print('valFGBG',seed,arm,scores['pooled']['macro_f1_present'],flush=True)
                del tail;gc.collect();torch.cuda.empty_cache()
            del template;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(0);template=SemanticTail(published,26).eval();corrected_full_interface(published,template,val,results)
        contrasts=comparisons(results);eligible=all(all(r['pooled_macro_difference']>0 for r in c['per_seed'])and c['mean_video_equal_difference']>0 for c in contrasts.values())
        for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
        features=json.loads((ROOT/'feature_manifest.json').read_text())
        for path,h in features['sha256'].items():assert ready.sha(path)==h,path
        atomic_json(ROOT/'summary.json',{'round':109,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'contrasts':contrasts,
            'SAM_FGBG_eligible_for_future_external_protocol':eligible,'geometry_main_weights_frozen':True,'geometry_truth_accuracy_evaluated':False,
            'all90_interface_checks_verified_in_run':True,'default_changed':False,'new_confirmation':False,'feature_manifest_sha256':ready.sha(ROOT/'feature_manifest.json'),
            'expanded_schedules_sha256':ready.sha(ROOT/'expanded_schedules.json'),'interface_checks_sha256':ready.sha(ROOT/'interface_checks.json'),
            'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}|{str(f):ready.sha(f)for f in ROOT.glob('*_trace.jsonl')}|{str(f):ready.sha(f)for f in ROOT.glob('*_validation.npz')}})
        print('DONE matched SAM foreground/background; eligibility',eligible,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True)
        atomic_json(ROOT/f'failure_{time.time_ns()}.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'outputs_preserved':True,'no_automatic_setting_or_label_relaxation':True})
        raise
