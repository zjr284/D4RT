"""Bounded temporary eight-step reproducibility diagnosis; candidates untouched."""
from pathlib import Path
import gc,json
import numpy as np
import torch
from semantic import d4rt_input_resolution as experiment
from semantic import d4rt_finetune_readiness as ready
from semantic.model import SemanticTail
from semantic.run import atomic_json
from tools.load_opend4rt import load_model
ROOT=experiment.ROOT

def main():
    originals={str(p):ready.sha(p)for p in ROOT.glob('*.pt')}
    originals|={str(p):ready.sha(p)for p in ROOT.glob('*_validation.npz')}
    input_files=['semantic/verify_d4rt_input_resolution.py','semantic_artifacts/d4rt_input_resolution_verification.log',
        'semantic/diagnose_resolution_replay.py','semantic/OVIS_D4RT_RESOLUTION_REPLAY_DIAGNOSIS.md',str(ROOT/'summary.json')]
    frozen={p:ready.sha(p)for p in input_files};atomic_json(ROOT/'replay_diagnosis_protocol.json',{'steps':8,'repeats':2,'originals_sha256':originals,'diagnostic_inputs_sha256':frozen})
    data=experiment.load_features('train');published=load_model(ready.CHECKPOINT,ready.CONFIG)
    initial=torch.load(experiment.PARENT/'initial_seed0.pt',map_location='cpu',weights_only=True)
    schedule=json.loads(Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json').read_text())['schedules']['0'][:8]
    trace=[json.loads(x)for x in (ROOT/'source_GT_seed0_trace.jsonl').read_text().splitlines()][:8]
    original_grad=torch.load(ROOT/'source_GT_seed0_step0.pt',map_location='cpu',weights_only=True)['gradients']
    states=[];rows=[];gradient_differences=[];operators=[]
    for repeat in range(2):
        torch.manual_seed(0);template=SemanticTail(published,26).eval();tail=experiment.make_tail(template,False,initial)
        opt=torch.optim.AdamW([{'params':[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)
        losses=[];runstates=[]
        for step,row in enumerate(schedule):
            g=data[row['GT_video']];a=data[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                go=tail(g['source'][gi][None],g['memory'][None])['semantic_logits'][0].float();ao=tail(a['source'][ai][None],a['memory'][None])['semantic_logits'][0].float()
            ce=torch.nn.functional.cross_entropy(go[:64],g['y'][gi[:64]]);sam=torch.nn.functional.cross_entropy(ao,a['y'][ai]);extra=torch.nn.functional.cross_entropy(go[64:],g['y'][gi[64:]])
            loss=ce+0.*sam+0.*extra;opt.zero_grad(set_to_none=True);loss.backward()
            if step==0:gradient_differences.append({n:float((q.grad.cpu()-original_grad[n]).abs().max())for n,q in tail.named_parameters()if q.grad is not None})
            opt.step();losses.append(float(ce.detach()));runstates.append(ready.weights(tail))
        states.append(runstates);rows.append({'repeat':repeat,'GT_CE':losses,'GT_CE_minus_original':[v-r['GT_CE']for v,r in zip(losses,trace)]})
        if repeat==0:
            opt.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):out=tail(g['source'][gi][None],g['memory'][None])['semantic_logits'][0].float()
                out.sum().backward()
            operators=[e.key for e in prof.key_averages()if any(w in e.key.lower()for w in ('attention','scaled_dot','cudnn'))]
        del tail,opt,template;gc.collect();torch.cuda.empty_cache()
    per_step=[]
    for i in range(8):
        differences={n:float((states[0][i][n]-states[1][i][n]).abs().max())for n in states[0][i]}
        per_step.append({'step':i+1,'GT_CE_repeat_difference':rows[0]['GT_CE'][i]-rows[1]['GT_CE'][i],'parameter_max_abs_difference':max(differences.values()),'differing_parameters':[n for n,d in differences.items()if d>0]})
    for path,h in originals.items():assert ready.sha(path)==h,path
    atomic_json(ROOT/'replay_diagnosis.json',{'runs':rows,'repeated_step_comparisons':per_step,'first_gradients_vs_original_max_abs':gradient_differences,
        'repeated_training_exact':all(r['parameter_max_abs_difference']==0 and r['GT_CE_repeat_difference']==0 for r in per_step),
        'operators':operators,'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_sdp_enabled':torch.backends.cuda.cudnn_sdp_enabled(),
        'original_outputs_unchanged':True,'diagnosis_protocol_sha256':ready.sha(ROOT/'replay_diagnosis_protocol.json')})
    print('DONE temporary replay diagnosis',rows,per_step,operators,flush=True)
if __name__=='__main__':main()
