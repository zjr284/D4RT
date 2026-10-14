"""Fixed BF16 encoder/tokens, promoted quantized-weight FP32 decoder control."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_native_interface import load_native_selected_adapter
from semantic.dino_native_rgb_control import ROOT as PREVIOUS,feature_path
from semantic.dino_query_stream_probe import ROOT as STREAM
from semantic.dino_interface import cuda_bf16
from semantic.dino_interface_probe import queries,measured,delta
from semantic.model import GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_decode_precision_probe')
SIZES=(1,8,32,64)
ARMS=('original_bf16','fixed_tokens_bf16','fixed_tokens_fp32')
PERMUTATIONS={'identity':np.arange(64),'reverse':np.arange(63,-1,-1),'even_then_odd':np.r_[np.arange(0,64,2),np.arange(1,64,2)]}


def dump(path,out):
    np.savez_compressed(path,**{k:v.float().cpu().numpy()for k,v in out.items()})


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed precision probe exists')
    prev=json.loads((PREVIOUS/'summary.json').read_text());stream=json.loads((STREAM/'summary.json').read_text())
    audit=json.loads((STREAM/'verification.json').read_text());assert audit['summary_sha256']==file_hash(STREAM/'summary.json')
    clips=prev['manifest']['confirmation'][:3];assert [c['video_id']for c in clips]==[345,574,573]
    paths=[STREAM/'summary.json',STREAM/'verification.json',Path('semantic/dino_decode_precision_probe.py'),Path('semantic/OVIS_DINO_DECODE_PRECISION_PROTOCOL.md')]
    hashes=dict(stream['protocol']['input_sha256']);hashes.update({str(p):file_hash(p)for p in paths})
    for path,h in hashes.items():assert file_hash(path)==h
    p={'round':28,'video_ids':[345,574,573],'B2_ids':[345,574],'indices':list(range(0,2048,32)),
       'arms':list(ARMS),'sizes':list(SIZES),'permutations':{k:v.tolist()for k,v in PERMUTATIONS.items()},'input_sha256':hashes,
       'limits':'Frozen quantized encoder/tokens/weights; FP32 arithmetic promotion, not original FP32 weights or truth geometry score'}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Protocol changed')
    atomic_json(pp,p);adapter=load_native_selected_adapter(0);base=adapter.base
    assert all(w.dtype==torch.bfloat16 for w in base.parameters())
    decoder=copy.deepcopy(base.decoder).float().eval();heads=copy.deepcopy(base.heads).float().eval()
    for left,right in [(base.decoder,decoder),(base.heads,heads)]:
        for a,b in zip(left.parameters(),right.parameters()):assert torch.equal(a.float(),b)
    batches=[]
    for c in clips:
        with np.load(feature_path('native_gt','confirmation',c['video_id']))as z:data={k:z[k].copy()for k in ['xy','t_tgt','t_cam']}
        batches.append({'video':load_video(c)[None].cuda().to(torch.bfloat16),'query':queries(data,np.arange(0,2048,32),extra=False)})
    pair={'video':torch.cat([b['video']for b in batches[:2]]),'query':{k:torch.cat([b['query'][k]for b in batches[:2]])for k in batches[0]['query']}}
    original_precision=torch.get_float32_matmul_precision();original_tf32=torch.backends.cuda.matmul.allow_tf32
    cases=[]
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            for kind,b,ids in [('B1',b,[c['video_id']])for b,c in zip(batches,clips)]+[('B2',pair,[345,574])]:
                with cuda_bf16(b['video'].device):
                    memory=base.encode_video(b['video']);q=b['query']
                    tokens=base.query_embedder(video=b['video'],u=q['u'].to(b['video'].dtype),v=q['v'].to(b['video'].dtype),
                         t_src=q['t_src'].long(),t_tgt=q['t_tgt'].long(),t_cam=q['t_cam'].long())
                mem32=memory.float();tok32=tokens.float()
                folder=ROOT/('_'.join(map(str,ids)));folder.mkdir(exist_ok=True)
                def run(arm,size,permutation):
                    parts=[];perm=torch.as_tensor(permutation,device='cuda')
                    for start in range(0,64,size):
                        ix=perm[start:start+size]
                        if arm=='fixed_tokens_fp32':
                            with torch.autocast('cuda',enabled=False):out=heads(decoder(tok32[:,ix].contiguous(),mem32))
                        else:
                            with cuda_bf16(b['video'].device):
                                if arm=='original_bf16':out=base.decode_queries(b['video'],{k:v[:,ix]for k,v in q.items()},memory)
                                else:out=base.heads(base.decoder(tokens[:,ix].contiguous(),memory))
                        assert set(out)==set(GEOMETRY_KEYS) and all(torch.isfinite(v).all()for v in out.values())
                        parts.append(out)
                    inverse=torch.argsort(perm)
                    return {k:torch.cat([o[k]for o in parts],dim=1)[:,inverse]for k in GEOMETRY_KEYS}
                refs={arm:run(arm,64,PERMUTATIONS['identity'])for arm in ARMS}
                assert all(torch.equal(refs[ARMS[0]][k],refs[ARMS[1]][k])for k in GEOMETRY_KEYS)
                for arm,r in refs.items():dump(folder/(arm+'_reference.npz'),r)
                records=[]
                for arm in ARMS:
                    for name,perm in PERMUTATIONS.items():
                        for size in SIZES:
                            out=run(arm,size,perm);path=folder/(arm+'_'+name+'_'+str(size)+'.npz');dump(path,out)
                            records.append({'arm':arm,'permutation':name,'size':size,'path':str(path),'sha256':file_hash(path),
                                'within_arm':{k:delta(out[k],refs[arm][k])for k in GEOMETRY_KEYS},
                                'vs_original_full64':{k:delta(out[k],refs[ARMS[0]][k])for k in GEOMETRY_KEYS}})
                timing=[]
                for repeat in range(3):
                    sizes=SIZES[repeat:]+SIZES[:repeat];arms=ARMS[repeat:]+ARMS[:repeat]
                    for arm in arms:
                        for size in sizes:
                            _,cost=measured(lambda:run(arm,size,PERMUTATIONS['identity']))
                            timing.append({'repetition':repeat,'arm':arm,'size':size,**cost})
                case={'kind':kind,'video_ids':ids,'tokens_dtype':str(tokens.dtype),'memory_dtype':str(memory.dtype),
                    'reference_paths':{arm:str(folder/(arm+'_reference.npz'))for arm in ARMS},
                    'reference_sha256':{arm:file_hash(folder/(arm+'_reference.npz'))for arm in ARMS},'records':records,'timings':timing}
                cases.append(case);atomic_json(ROOT/'progress.json',{'protocol_sha256':file_hash(pp),'cases':cases})
                print('precision completed',kind,ids,'max within-arm xyz',
                      {arm:max(r['within_arm']['xyz_3d']['max_absolute_difference']for r in records if r['arm']==arm)for arm in ARMS},flush=True)
    finally:
        torch.set_float32_matmul_precision(original_precision);torch.backends.cuda.matmul.allow_tf32=original_tf32
    for path,h in hashes.items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'cases':cases,'verification':{'original_fixed_tokens_full64_bitwise_equal':True,
       'fp32_weights_exact_promotion_of_bf16_values':True,'all_inputs_unchanged':True,'precision_settings_restored':True,'no_training_or_new_video_scores':True},
       'original_precision_settings':{'float32_matmul_precision':original_precision,'allow_tf32':original_tf32}})
    print('DONE fixed-token decoder precision control',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
