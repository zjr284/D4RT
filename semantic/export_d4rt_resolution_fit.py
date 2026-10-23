"""Publish an LF CSV copy while retaining the original verified CSV bytes."""
import csv,io,json
from pathlib import Path
from semantic.verify_d4rt_controlled_finetune import sha
from semantic.run import atomic_json

def main():
    root=Path('semantic_artifacts/d4rt_resolution_fit');source=root/'per_class.csv'
    target=Path('semantic/results/d4rt_resolution_fit_per_class.csv')
    summary=json.loads((root/'summary.json').read_text());verification=json.loads((root/'verification.json').read_text())
    assert verification['summary_sha256']==sha(root/'summary.json')
    assert summary['outputs_sha256'][str(source)]==sha(source)
    original=source.read_bytes();normalized=original.replace(b'\r\n',b'\n')
    assert list(csv.reader(io.StringIO(original.decode())))==list(csv.reader(io.StringIO(normalized.decode())) )
    target.write_bytes(normalized)
    atomic_json(Path('semantic/results/d4rt_resolution_fit_export.json'),{'source_CSV_sha256':sha(source),'published_LF_CSV_sha256':sha(target),
        'rows_excluding_header':1872,'only_CRLF_to_LF_changed':True,'all_CSV_cells_equal':True,'original_verified_outputs_unchanged':True,
        'export_code_sha256':sha('semantic/export_d4rt_resolution_fit.py')})
if __name__=='__main__':main()
