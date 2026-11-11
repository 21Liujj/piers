"""Shared request contracts, panorama instance association and output artifacts.

No weights or GPU imports here. Association is an engineering heuristic, not a
claim of instance AP. It requires overlap evidence and respects periodic width.
"""
from __future__ import annotations
import colorsys, hashlib, io, json, math, re, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERSION = 'panorama-workbench-20261111-v2'
SOURCE_CLASSES = ['road','sidewalk','building','wall','fence','pole','traffic light',
                  'traffic sign','vegetation','terrain','sky','person','car']
STREET_CLASSES = SOURCE_CLASSES + ['curb','crosswalk','truck','bus','bicycle','motorcycle','train','rider']
INDOOR_CLASSES = ['ceiling','chair','door','floor','sofa','table','wall','window']
ZH = {'road':'道路','sidewalk':'人行道','building':'建筑','wall':'墙面','fence':'围栏','pole':'杆柱',
      'traffic light':'交通灯','traffic sign':'交通标志','vegetation':'植被','terrain':'地形','sky':'天空',
      'person':'行人','car':'汽车','curb':'路缘','crosswalk':'人行横道','truck':'卡车','bus':'公交车',
      'bicycle':'自行车','motorcycle':'摩托车','train':'火车','rider':'骑行者','ceiling':'天花板',
      'chair':'椅子','door':'门','floor':'地板','sofa':'沙发','table':'桌子','window':'窗户'}
ALIASES_ZH = {v:k for k,v in ZH.items()}
ALIASES_ZH.update({'车':'car','小汽车':'car','轿车':'car','货车':'truck','大巴':'bus','人':'person',
                   '人群':'person','树':'tree','树木':'tree','路灯':'street lamp','墙':'wall',
                   '窗':'window','桌':'table','椅':'chair','红绿灯':'traffic light','斑马线':'crosswalk'})
COLORS = ['#65b4ff','#e3bd65','#ab9ff6','#dd8fac','#d98c66','#82bdc8','#f18c8c',
          '#e6a1dd','#8ccf86','#b0c56b','#8bceeb','#ff9a6a','#5793f3','#a5a3a3',
          '#f0d89a','#b87dba','#f39b82','#72c7b2','#c7ac79','#f4799d','#73a8ae']
MAX_UPLOAD = 30 * 1024 * 1024
MAX_PIXELS = 36_000_000
JOB_ID = re.compile(r'^[a-f0-9]{24}$')
COUNTABLE = {'car','person','truck','bus','bicycle','motorcycle','train','rider','chair','table',
             'sofa','door','window','traffic light','traffic sign','pole','tree','street lamp'}

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def valid_job_id(value):
    if not JOB_ID.fullmatch(value):raise ValueError('无效的任务编号')
    return value

def class_color(index):return COLORS[index % len(COLORS)]

def instance_color(index):
    rgb=colorsys.hsv_to_rgb((index*.618033988749895+.08)%1,.54,.96)
    return '#'+bytes(round(c*255) for c in rgb).hex()

def parse_classes(text, scene='street'):
    if scene not in ('street','indoor'):raise ValueError('请选择街景或室内场景')
    parts=[x.strip() for x in re.split(r'[,，;；\n、]+',text) if x.strip()]
    if not parts:return list(STREET_CLASSES if scene=='street' else INDOOR_CLASSES),False
    result=[]
    concepts={'cars':'car','people':'person','pedestrians':'person','pedestrian':'person',
              'vehicles':'car','vehicle':'car','side walk':'sidewalk','trafficlight':'traffic light'}
    seen_concepts={}
    for part in parts:
        if len(part)>100:raise ValueError('单个目标描述请控制在100字以内')
        name=ALIASES_ZH.get(part,' '.join(part.lower().replace('_',' ').split()))
        if re.search(r'[\u4e00-\u9fff]',name):
            raise ValueError('这个中文描述暂未收录，请用英文类别或短语，例如 red car；常见中文类别可直接使用')
        if not name or any(ord(c)<32 for c in name):raise ValueError('目标描述包含无效字符')
        concept=concepts.get(name,name)
        if concept in seen_concepts and seen_concepts[concept]!=name:
            raise ValueError('类别含有重叠概念：'+seen_concepts[concept]+' / '+name+'，请保留一个名称')
        seen_concepts[concept]=name
        if name not in result:result.append(name)
    if len(result)>32:raise ValueError('一次最多指定32个目标类别')
    return result,True

def decode_image(raw):
    if not raw or len(raw)>MAX_UPLOAD:raise ValueError('请选择30MB以内的图片')
    try:
        with Image.open(io.BytesIO(raw)) as src:
            if src.width*src.height>MAX_PIXELS:raise ValueError('图片超过3600万像素，请缩小后上传')
            if min(src.size)<64:raise ValueError('图片的宽和高至少为64像素')
            return ImageOps.exif_transpose(src).convert('RGB').copy()
    except (OSError,Image.DecompressionBombError) as exc:
        raise ValueError('无法读取图片，请上传有效的 PNG、JPEG 或 WebP') from exc

def canonical_image(image, scene):
    # Outdoor geometry is the frozen paper protocol; indoor is a separate
    # text-only generic route until adapted weights have passed validation.
    return image.resize((2048,400 if scene=='street' else 1024),Image.Resampling.BICUBIC)

def query_mapping(meta, classes, scene):
    if scene=='indoor':return {q['query_id']:i for i,q in enumerate(meta['queries'])}
    # Preserve literal + canonical query support from the published v3 runtime.
    from full_model_queries_v3 import make_queries,class_by_query_id
    _,bank,output,_=make_queries(meta['image_id'],classes)
    mapping=class_by_query_id(classes,bank,output)
    return {qid:classes.index(name) for qid,name in mapping.items()}

def group_instances(meta, masks, classes, mapping, guided=None):
    """Associate the same object across FoVs, including the wraparound FoV.

    Shared-visible IoU >= .55 is required. Same-view duplicate queries require
    .85 global IoU. Disjoint fragments are never guessed to be one instance.
    A group's match is checked against a member, avoiding union-mask bridges.
    """
    h,w=meta['shape'];valids={}
    for f in meta['fovs']:
        v=np.zeros((h,w),bool);cols=(f['x_start']+np.arange(f['win_w']))%w
        v[:,cols]=True;valids[f['view_id']]=v
    items=[]
    for n,c in enumerate(meta['candidates']):
        if c['query_id'] not in mapping:continue
        idx=mapping[c['query_id']];mask=np.asarray(masks[n],bool).copy()
        if guided is not None:mask &= guided[idx]
        if np.count_nonzero(mask)<12:continue
        items.append({'class_index':idx,'mask':mask,'view_id':c['fov_id'],'valid':valids[c['fov_id']],
                      'candidate_id':c['id'],'detector_score':float(c['detector_score']),
                      'sam_quality':float(c['sam_quality']),
                      'score':max(0,float(c['detector_score']))*max(0,float(c['sam_quality']))})
    groups=[]
    for c in sorted(items,key=lambda x:(-x['score'],x['candidate_id'])):
        target=None;best=.55
        for group in groups:
            if group['class_index']!=c['class_index']:continue
            match=0.
            for d in group['members']:
                valid=c['valid'] & d['valid']
                a=c['mask'] & valid;b=d['mask'] & valid
                inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b)
                if inter<12:continue
                iou=inter/max(union,1)
                if c['view_id']==d['view_id']:
                    global_iou=np.count_nonzero(c['mask']&d['mask'])/max(np.count_nonzero(c['mask']|d['mask']),1)
                    if global_iou<.85:continue
                match=max(match,iou)
            if match>=best:target=group;best=match
        if target is None:groups.append({'class_index':c['class_index'],'mask':c['mask'].copy(),'members':[c],'score':c['score']})
        else:target['mask'] |= c['mask'];target['members'].append(c)
    return groups

def periodic_box_segments(mask):
    """Return a shortest circular bounding interval, split at image width."""
    ys,xs=np.nonzero(mask)
    if not len(xs):return []
    w=mask.shape[1];cols=np.unique(xs);gaps=np.diff(np.r_[cols,cols[0]+w]);cut=int(np.argmax(gaps))
    start=int(cols[(cut+1)%len(cols)]);end=int(cols[cut])+1
    y1=int(ys.min());y2=int(ys.max())+1
    if start<end:return [[start,y1,end,y2]]
    return [[start,y1,w,y2],[0,y1,end,y2]]

def build_artifacts(folder, image, classes, meta, masks, mapping, guided, prediction, mode, audit):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    w,h=image.size;ch,cw=meta['shape'];groups=group_instances(meta,masks,classes,mapping,guided)
    if mode=='text':
        prediction=np.full((ch,cw),255,np.uint8)
        # Guided category union is authoritative; instance association cannot
        # introduce pixels. Stable class order resolves cross-query overlaps.
        for index in reversed(range(len(classes))):prediction[guided[index]]=index
    label=np.asarray(Image.fromarray(prediction.astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST))
    Image.fromarray(label).save(folder/'classes.png')
    rgb=np.asarray(image).copy();colors=np.array([tuple(bytes.fromhex(class_color(i)[1:])) for i in range(len(classes))],np.uint8)
    active=label!=255;class_rgb=np.zeros_like(rgb);class_rgb[active]=colors[label[active]]
    rgba=np.zeros((h,w,4),np.uint8);rgba[:,:,:3]=class_rgb;rgba[:,:,3]=active.astype(np.uint8)*255
    Image.fromarray(rgba).save(folder/'semantic.png')
    overlay=rgb.copy();overlay[active]=(rgb[active]*.48+class_rgb[active]*.52).astype(np.uint8)
    Image.fromarray(overlay).save(folder/'overlay.jpg',quality=93)
    # Assign each pixel to one global object; deterministic priority, then
    # assign compact IDs only to objects with a nonempty visible mask.
    instance_map=np.zeros((h,w),np.uint16);records=[];counts={}
    stuff=(set(SOURCE_CLASSES)-COUNTABLE)|{'floor','ceiling','ground','grass','water','curb','crosswalk'}
    selected=[g for g in groups if classes[g['class_index']] not in stuff]
    for g in sorted(selected,key=lambda x:-x['score']):
        m=np.asarray(Image.fromarray(g['mask']).resize((w,h),Image.Resampling.NEAREST),bool).copy()
        if mode!='text':m &= label==g['class_index']
        m &= instance_map==0
        if np.count_nonzero(m)<12:continue
        iid=len(records)+1;instance_map[m]=iid;idx=g['class_index'];counts[idx]=counts.get(idx,0)+1
        records.append({'id':iid,'class_index':idx,'class_name':classes[idx],
                        'label':classes[idx]+' '+str(counts[idx]),'pixels':int(m.sum()),
                        'box_segments':periodic_box_segments(m),'detector_score':max(c['detector_score'] for c in g['members']),
                        'sam_quality':max(c['sam_quality'] for c in g['members']),
                        'views':sorted({c['view_id'] for c in g['members']}),
                        'source_candidates':[c['candidate_id'] for c in g['members']]})
    Image.fromarray(instance_map).save(folder/'instances.png')
    inst_rgba=np.zeros((h,w,4),np.uint8);instance_overlay=overlay.copy()
    for r in records:
        m=instance_map==r['id'];color=tuple(bytes.fromhex(instance_color(r['id'])[1:]))
        r['color']='#'+bytes(color).hex();inst_rgba[m]=(*color,255)
        instance_overlay[m]=(rgb[m]*.35+np.array(color)*.65).astype(np.uint8)
    Image.fromarray(inst_rgba).save(folder/'instance_layer.png')
    draw_image=Image.fromarray(instance_overlay);draw=ImageDraw.Draw(draw_image)
    try:font=ImageFont.truetype('DejaVuSans.ttf',max(14,round(w/80)))
    except OSError:font=ImageFont.load_default()
    for r in records:
        segments=r['box_segments']
        for box in segments:draw.rectangle(box,outline=r['color'],width=max(1,w//800))
        if segments:
            box=max(segments,key=lambda b:b[2]-b[0]);x,y=box[:2];y=max(0,y-20)
            bounds=draw.textbbox((x,y),r['label'],font=font)
            draw.rectangle((bounds[0]-3,bounds[1]-3,bounds[2]+3,bounds[3]+3),fill='#16252c')
            draw.text((x,y),r['label'],fill='white',font=font)
    draw_image.save(folder/'instance_overlay.jpg',quality=93)
    rows=[]
    for idx,name in enumerate(classes):
        area=int(np.count_nonzero(label==idx))
        rows.append({'index':idx,'name':name,'name_zh':ZH.get(name,name),'color':class_color(idx),
                     'pixels':area,'fraction':area/(w*h),'instances':counts.get(idx,0)})
    result={'version':VERSION,'size':[w,h],'mode':mode,'classes':rows,'instances':records,
            'instance_count':len(records),'foreground_fraction':float(active.mean()),'audit':audit,
            'accuracy':None,'accuracy_note':'无标准标签，未计算准确率；检测分数和SAM质量均不是IoU',
            'instance_protocol':{'version':'shared-visible-iou-v2','iou_threshold':.55,'same_view_duplicate_iou':.85,
                                 'stuff_excluded':sorted(stuff),
                                 'disjoint_fragments_merged':False,'instance_AP_measured':False},
            'files':['classes.png','semantic.png','overlay.jpg','instances.png','instance_layer.png','instance_overlay.jpg','result.json']}
    write_json(folder/'result.json',result)
    with zipfile.ZipFile(folder/'results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in result['files']:z.write(folder/name,name)
    return result

def score_uploaded_gt(prediction, truth, classes, encoding='class_ids'):
    p=np.asarray(prediction);g=np.asarray(truth)
    if g.ndim!=2 or p.shape!=g.shape:raise ValueError('标注需为单通道 PNG，尺寸与原图一致')
    n=len(classes)
    if encoding=='binary':
        if n!=1:raise ValueError('二值标注仅适用于单个文本目标；多类别请使用类别ID标注')
        if not set(np.unique(g)).issubset({0,1,255}):raise ValueError('二值标注只支持0背景、1或255目标')
        foreground=g>0;found=p==0
        tp=int((foreground&found).sum());fp=int((~foreground&found).sum());fn=int((foreground&~found).sum())
        den=tp+fp+fn;iou=100*tp/den if den else None;dice=200*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None
        return {'scope':'current_image','encoding':'binary','mIoU_pct':iou,'mDice_pct':dice,
                'pixel_accuracy_pct':100*float((foreground==found).mean()),'per_class_IoU_pct':[iou],
                'classes':classes,'valid_pixels':int(g.size),'correct_empty':den==0,'TP':tp,'FP':fp,'FN':fn}
    if encoding!='class_ids':raise ValueError('不支持的标注编码')
    valid=g!=255
    if not valid.any():raise ValueError('标注全部为忽略像素255，无法计算准确率')
    if ((g[valid]<0)|(g[valid]>=n)).any():raise ValueError('标注类别ID应与本次图例的0..N-1一致，255为忽略')
    pp=np.where((p[valid]>=0)&(p[valid]<n),p[valid],n).astype(np.int64)
    hist=np.bincount(g[valid].astype(np.int64)*(n+1)+pp,minlength=n*(n+1)).reshape(n,n+1)
    tp=np.diag(hist[:,:n]);gt=hist.sum(1);pr=hist[:,:n].sum(0);union=gt+pr-tp
    def ratio(a,b):return np.divide(a,b,out=np.full(n,np.nan),where=b>0)
    iou=ratio(tp*100,union);dice=ratio(tp*200,gt+pr);acc=ratio(tp*100,gt)
    clean=lambda v:[float(x) if np.isfinite(x) else None for x in v]
    return {'scope':'current_image','encoding':'class_ids','mIoU_pct':float(np.nanmean(iou)),
            'mDice_pct':float(np.nanmean(dice)),'mAcc_pct':float(np.nanmean(acc)),
            'pixel_accuracy_pct':float(tp.sum()/hist.sum()*100),
            'FWIoU_pct':float(np.nansum((gt/gt.sum())*iou)), 'per_class_IoU_pct':clean(iou),
            'classes':classes,'confusion_matrix':hist.tolist(),'valid_pixels':int(valid.sum()),
            'note':'单图指标；数据集mIoU须汇总混淆矩阵后计算'}
