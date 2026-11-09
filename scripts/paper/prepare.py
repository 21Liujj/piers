"""Validate exact dataset samples and make a GT-free inference manifest."""
from pathlib import Path
import hashlib,json,os

REPO=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(task,out,data=None,extra=None,limit=0):
    import numpy as np
    from PIL import Image
    tasks=json.loads((REPO/'configs/piers/paper20261109/tasks.json').read_text());spec=tasks[task]
    samples=json.loads((REPO/'configs/piers/paper20261109/datasets'/spec['samples']).read_text())['samples']
    if limit:samples=samples[:limit]
    roots={'data':Path(data or os.environ.get('PIERS_DATA_ROOT',str(REPO/'data'))).resolve(),
           'extra':Path(extra or os.environ.get('PIERS_DATA_EXTRA_ROOT',str(REPO/'data_extra'))).resolve()}
    out=Path(out).resolve();manifest=[];index=[];pixel_count=np.zeros(len(spec['classes']),np.int64)
    if (out/'manifest.json').exists():raise ValueError('Use a new prepare directory; never silently reuse changed inputs')
    def path(item):
        result=(roots[item['root']]/item['path']).resolve()
        if not result.is_relative_to(roots[item['root']]):raise ValueError('Dataset path escapes root')
        return result
    lut=None
    if 'map' in spec:lut=np.full(256,255,np.uint8);lut[:len(spec['map'])]=spec['map']
    for sample in samples:
        options=[sample['source']]+sample.get('source_alternatives',[])
        source=next((path(item) for item in options if path(item).is_file()),None)
        if source is None:raise FileNotFoundError('RGB missing: '+str([str(path(x)) for x in options]))
        gt=path(sample['GT'])
        if not gt.is_file():raise FileNotFoundError('GT missing: '+str(gt))
        if sha(gt)!=sample['GT_sha256']:raise ValueError('GT checksum differs: '+str(gt))
        with Image.open(source) as image:
            size=list(image.size)
            if size!=sample['original_size']:raise ValueError('Source geometry differs: '+str(source))
            canonical=image.convert('RGB').resize((2048,400),Image.Resampling.BICUBIC)
            if hashlib.sha256(canonical.tobytes()).hexdigest()!=sample['canonical_pixel_sha256']:
                raise ValueError('Model input pixels differ from recorded experiment: '+str(source))
        destination=out/'images'/(sample['image_id']+'.png');destination.parent.mkdir(parents=True,exist_ok=True);canonical.save(destination)
        with Image.open(gt) as image:label=np.asarray(image)
        if label.shape!=(size[1],size[0]):raise ValueError('GT/RGB dimensions differ')
        if lut is not None:
            if not set(np.unique(label))<=set(range(len(spec['map'])))|{255}:raise ValueError('Unexpected pre-mapping GT ID')
            label=lut[label]
        values=set(np.unique(label).tolist())
        if not values<=set(spec['raw_ids'])|set(spec['ignore_raw_ids']):raise ValueError('Unexpected GT IDs')
        pixel_count += [int(np.count_nonzero(label==raw)) for raw in spec['raw_ids']]
        digest=sha(destination)
        manifest.append({'image_id':sample['image_id'],'path':str(destination),'width':2048,'height':400,
            'original_size':size,'classes':spec['classes'],'input_sha256':digest,'source_sha256':sha(source)})
        index.append({'image_id':sample['image_id'],'GT_path':str(gt),'GT_sha256':sample['GT_sha256']})
    write(out/'manifest.json',{'images':manifest})
    write(out/'SCORER_ONLY_INDEX.json',{'task':task,**spec,'images':index,'GT_pixels_per_class':pixel_count.tolist()})
    write(out/'PREPARATION.json',{'status':'complete','task':task,'images':len(samples),'canonical_pixel_hashes_verified':True,
        'manifest_sha256':sha(out/'manifest.json'),'GT_index_sha256':sha(out/'SCORER_ONLY_INDEX.json'),'original_GT_files_modified':False})
    return out/'manifest.json'
