"""Complete image+category interface, with real I1/I2/I3 and v2 OV competition.

Example:
  python scripts/piers/infer_complete_model.py --image street.png \
    --classes road sidewalk car truck bus curb crosswalk --out runs/user_case

No dataset name, GT, or target adaptation is accepted. Historical standalone
infer_open_vocab.py is retained for reproduction; this is the full-model API.
"""
from __future__ import annotations
import argparse,json,sys,time,hashlib
from pathlib import Path
from benchmark_complete_model_v3 import Service,write_json


def main(a):
    from PIL import Image
    from full_model_queries_v3 import make_queries
    repo=Path(a.repo).resolve();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output folder')
    classes=a.classes
    if a.classes_file:
        if classes:raise ValueError('Use --classes or --classes-file')
        classes=json.loads(Path(a.classes_file).read_text(encoding='utf-8'))
    if not isinstance(classes,list) or not classes or not all(isinstance(c,str) and c.strip() for c in classes):
        raise ValueError('Supply a nonempty list of category names')
    make_queries('input',classes)
    out.mkdir(parents=True,exist_ok=True);source=Path(a.image).resolve()
    with Image.open(source) as image:
        width,height=image.size;canonical=image.convert('RGB').resize((2048,400),Image.Resampling.BICUBIC)
    input_path=out/'input.png';canonical.save(input_path)
    row={'image_id':'input','path':str(input_path),'width':2048,'height':400,
         'original_size':[width,height],'classes':classes,
         'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    a.repo=str(repo);a.out=str(out)
    if not a.calibrators:
        a.calibrators=str(repo/'assets/paper20261109/calibrators')
    services=[]
    try:
        semantic=Service(a,a.method);services.append(semantic)
        text=Service(a,'text') if a.method=='full' else None
        if text:services.append(text)
        started=time.perf_counter()
        text_result=text.call('predict',row=row,folder=str(out/'text/input')) if text else None
        result=semantic.call('predict',row=row,text_parent=str(out/'text'),prediction=str(out/'prediction.png'),
                             verify_common13=False)
        report={'status':'complete','version':'complete-I1-I2-I3-OV-v3-20261109',
                'method':a.method,'class_names':classes,'image':str(source),
                'input_sha256':row['source_sha256'],'output_WH':[width,height],
                'prediction':str(out/'prediction.png'),'semantic':result,'text':text_result,
                'dataset_identity_input':False,'GT_reads':0,'weights_updated':False,
                'wall_seconds_after_loading':time.perf_counter()-started,
                'timing_note':'Single invocation; not a controlled isolated efficiency benchmark',
                'limitations':'Category-name open vocabulary; arbitrary words are accepted without guaranteed accuracy or referring-expression understanding'}
        write_json(out/'inference.json',report)
        print(json.dumps({'status':'complete','classes':classes,'prediction':str(out/'prediction.png')},ensure_ascii=False))
    finally:
        for service in reversed(services):service.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--classes',nargs='+');parser.add_argument('--classes-file')
    parser.add_argument('--method',choices=['original','full'],default='full')
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]));parser.add_argument('--calibrators')
    main(parser.parse_args())
