"""Documented import-only recovery of the immutable R77 frozen experiment."""
import inspect,json
from pathlib import Path
from torch.nn import functional as F
from semantic import coco_uniform_queries as experiment
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 root=experiment.ROOT
 if(root/'summary.json').exists():raise RuntimeError('Completed diagnostic exists; do not repeat')
 failure=json.loads((root/'failure.json').read_text());assert failure=={'type':'NameError','message':"name 'F' is not defined"}
 assert not list((root/'logits').glob('*.npz'))
 protocol=json.loads((root/'protocol.json').read_text())
 for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
 amendment={'scope':'import alias only; restore intended torch.nn.functional.linear before first logits','original_protocol_sha256':file_hash(root/'protocol.json'),'original_frozen_source_sha256':file_hash('semantic/coco_uniform_queries.py'),'initial_failure_sha256':file_hash(root/'failure.json'),'recovery_source_sha256':file_hash('semantic/coco_uniform_queries_resume.py'),'torch_functional_sha256':file_hash(inspect.getfile(F)),'query_sha256':{str(x):file_hash(x)for x in sorted((root/'queries').glob('*.npz'))},'feature_sha256':{str(x):file_hash(x)for x in sorted((root/'features').rglob('*.npz'))},'logits_before_recovery':0,'training_or_calibration_updates':0}
 assert len(amendment['query_sha256'])==80 and len(amendment['feature_sha256'])==160
 path=root/'import_amendment.json'
 if path.exists():assert json.loads(path.read_text())==amendment
 else:atomic_json(path,amendment)
 experiment.F=F
 experiment.main()
 summary=json.loads((root/'summary.json').read_text());summary['runtime_import_amendment_sha256']=file_hash(path);atomic_json(root/'summary.json',summary)
 print('DONE explicit import-only recovery; original frozen protocol/source preserved',flush=True)
if __name__=='__main__':main()
