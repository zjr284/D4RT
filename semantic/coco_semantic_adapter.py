"""Official COCO polygon/RLE conversion; two explicitly scoped semantic policies."""
from semantic.external_coco_adapter import label_map
from semantic.external_label_sensitivity import consensus_map

def semantic_maps(api,image,mapping,size=128):
 annotations=[dict(a,segmentation=api.annToRLE(a))for a in api.imgToAnns[image['id']]]
 strict=label_map(annotations,mapping,image['height'],image['width'],size)
 consensus=consensus_map(annotations,mapping,image['height'],image['width'],size)
 return {'strict':strict,'consensus':consensus}
