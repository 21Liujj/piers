"""CPU contract tests. Synthetic masks test geometry, never stand in for inference."""
import io,json,os,sys,tempfile,time,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
os.environ['PIERS_WEB_DISABLE_CONNECT']='1'
from common import (parse_classes,group_instances,periodic_box_segments,score_uploaded_gt,
                    build_artifacts,decode_image,STREET_CLASSES)
from app import create_app
from fastapi.testclient import TestClient
from gpu_guard import observed_gpu_pids,orchestrates_gpu

class GeometryContracts(unittest.TestCase):
    def candidates(self):
        h,w=8,32;mask=np.zeros((h,w),bool);mask[:,:4]=True;mask[:,28:]=True
        visible=np.zeros_like(mask);visible[:,:4]=True
        adjacent=np.zeros_like(mask);adjacent[:,6:10]=True
        meta={'image_id':'test','shape':[h,w],'queries':[{'query_id':'q'}],
              'fovs':[{'view_id':'wrap','x_start':24,'win_w':16},{'view_id':'left','x_start':0,'win_w':16}],
              'candidates':[{'id':'a','query_id':'q','fov_id':'wrap','detector_score':.9,'sam_quality':.9},
                            {'id':'b','query_id':'q','fov_id':'left','detector_score':.85,'sam_quality':.9},
                            {'id':'c','query_id':'q','fov_id':'left','detector_score':.8,'sam_quality':.9}]}
        return meta,np.stack([mask,visible,adjacent])
    def test_wrap_duplicate_is_one_but_adjacent_car_stays_separate(self):
        meta,masks=self.candidates();groups=group_instances(meta,masks,['car'],{'q':0})
        self.assertEqual(len(groups),2)
        self.assertEqual(sorted(len(g['members']) for g in groups),[1,2])
        segments=periodic_box_segments(groups[0]['mask'])
        self.assertEqual(segments,[[28,0,32,8],[0,0,4,8]])
    def test_disjoint_detections_never_guessed_to_be_one(self):
        meta,masks=self.candidates();masks[1]=False;masks[1,:,12:16]=True
        groups=group_instances(meta,masks,['car'],{'q':0});self.assertEqual(len(groups),3)
    def test_identical_shapes_from_different_queries_stay_separate(self):
        meta,masks=self.candidates();meta['candidates'][1]['query_id']='q2'
        groups=group_instances(meta,masks,['car','bus'],{'q':0,'q2':1})
        self.assertEqual(len(groups),3)
    def test_artifacts_keep_global_ids_and_query_background(self):
        meta,masks=self.candidates();guided=np.array([masks.any(0)])
        with tempfile.TemporaryDirectory() as tmp:
            r=build_artifacts(tmp,Image.new('RGB',(32,8)),['car'],meta,masks,{'q':0},guided,
                              np.full((8,32),255,np.uint8),'text',{'test_fixture':True})
            self.assertEqual(r['instance_count'],2)
            ids=np.asarray(Image.open(Path(tmp)/'instances.png'))
            self.assertEqual(set(np.unique(ids)),{0,1,2})
            self.assertEqual(ids[0,0],ids[0,31]);self.assertNotEqual(ids[0,0],ids[0,7])
            pred=np.asarray(Image.open(Path(tmp)/'classes.png'))
            self.assertEqual(pred[0,20],255)
            self.assertIsNone(r['accuracy'])
            with zipfile.ZipFile(Path(tmp)/'results.zip') as z:self.assertIn('instances.png',z.namelist())
    def test_floor_is_semantic_region_without_object_count(self):
        meta,masks=self.candidates();guided=np.array([masks.any(0)])
        with tempfile.TemporaryDirectory() as tmp:
            r=build_artifacts(tmp,Image.new('RGB',(32,8)),['floor'],meta,masks,{'q':0},guided,
                              np.full((8,32),255,np.uint8),'text',{'test_fixture':True})
            self.assertGreater(r['classes'][0]['pixels'],0);self.assertEqual(r['instance_count'],0)

class MetricContracts(unittest.TestCase):
    def test_open_vocabulary_default_and_chinese_aliases(self):
        names,text=parse_classes('','street');self.assertFalse(text);self.assertGreater(len(names),13)
        self.assertTrue({'curb','truck','bus','crosswalk'}<=set(names))
        self.assertEqual(parse_classes('汽车，person, car；红绿灯')[0],['car','person','traffic light'])
        self.assertEqual(parse_classes('traffic cone, purple backpack')[0],['traffic cone','purple backpack'])
        self.assertEqual(parse_classes('vehicles')[0],['vehicles'])
        with self.assertRaises(ValueError):parse_classes('car, vehicles')
        self.assertEqual(len(parse_classes('','indoor')[0]),8)
    def test_unsupported_chinese_does_not_silently_change_request(self):
        with self.assertRaises(ValueError):parse_classes('最左边那辆紫色的小车')
    def test_false_positive_on_unknown_background_counts(self):
        p=np.array([[0,0],[255,255]],np.uint8);g=np.array([[255,0],[0,0]],np.uint8)
        m=score_uploaded_gt(p,g,['car'])
        self.assertAlmostEqual(m['mIoU_pct'],100/3)
        b=score_uploaded_gt(p,np.array([[0,255],[255,0]],np.uint8),['car'],'binary')
        self.assertAlmostEqual(b['mIoU_pct'],100/3)
    def test_empty_binary_is_unscorable_iou_and_correct_reject(self):
        m=score_uploaded_gt(np.full((4,4),255,np.uint8),np.zeros((4,4),np.uint8),['car'],'binary')
        self.assertIsNone(m['mIoU_pct']);self.assertTrue(m['correct_empty'])
    def test_invalid_labels_cannot_generate_accuracy(self):
        p=np.zeros((4,4),np.uint8)
        for gt in [np.full((4,4),255,np.uint8),np.full((4,4),22,np.uint8),np.zeros((3,4),np.uint8)]:
            with self.assertRaises(ValueError):score_uploaded_gt(p,gt,['car'])

class GPUCoordinationContracts(unittest.TestCase):
    def test_empty_nvml_does_not_hide_experiment_or_device_owner(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            for pid,argv in [('10',['python','scripts/paper/reproduce.py','run']),('11',['python','custom.py']),('12',['python','cpu_analysis.py'])]:
                folder=root/pid;(folder/'fd').mkdir(parents=True);(folder/'cmdline').write_bytes('\0'.join(argv).encode());(folder/'fd/3').touch()
            with patch('gpu_guard.os.readlink',side_effect=lambda p:'/dev/nvidia0' if str(p.parent.parent.name)=='11' else '/tmp/log'):
                self.assertEqual(observed_gpu_pids(root,nvml_pids=[]),{10,11})
    def test_ledger_replay_does_not_block_gpu(self):
        self.assertFalse(orchestrates_gpu(['python','reproduce.py','verify-ledgers']))

class OfflineBridge:
    def public_config(self):return {'host':'test','port':22,'username':'test','has_credentials':False}
    def health(self,*a,**kw):return {'connected':False,'message':'not configured'}
    def connect(self,*a,**kw):raise RuntimeError('not configured')
    def sanitize_error(self,exc):return str(exc)
    def close(self):pass

class APIContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.app=create_app(self.tmp.name,OfflineBridge())
        self.client=TestClient(self.app);self.client.__enter__()
    def tearDown(self):self.client.__exit__(None,None,None);self.tmp.cleanup()
    def image(self):
        out=io.BytesIO();Image.new('RGB',(128,64)).save(out,format='PNG')
        r=self.client.post('/api/images',files={'file':('image.png',out.getvalue(),'image/png')})
        self.assertEqual(r.status_code,200);return r.json()
    def test_empty_text_is_supported_and_not_limited_to13(self):
        im=self.image();r=self.client.post('/api/jobs',json={'image_id':im['id'],'scene':'street','text':''})
        self.assertEqual(r.status_code,202);self.assertEqual(r.json()['mode'],'semantic')
        self.assertEqual(r.json()['classes'],STREET_CLASSES)
    def test_bad_upload_and_unknown_image_are_rejected(self):
        self.assertEqual(self.client.post('/api/images',files={'file':('bad.png',b'not image','image/png')}).status_code,400)
        self.assertEqual(self.client.post('/api/jobs',json={'image_id':'a'*24,'text':'car'}).status_code,404)
    def test_cross_origin_posts_rejected_and_no_gt_score_before_result(self):
        im=self.image()
        r=self.client.post('/api/jobs',headers={'Origin':'https://unrelated.example'},json={'image_id':im['id']})
        self.assertEqual(r.status_code,403)
        r=self.client.post('/api/jobs',json={'image_id':im['id'],'text':'car'});jid=r.json()['id']
        r=self.client.post('/api/jobs/'+jid+'/evaluate',files={'file':('gt.png',b'','image/png')})
        self.assertEqual(r.status_code,409)
    def test_archived_numbers_match_repo_not_current_upload(self):
        r=self.client.get('/api/experiments');self.assertEqual(r.status_code,200)
        wild=r.json()['rows'][1];self.assertEqual(wild['images'],500)
        self.assertAlmostEqual(wild['full_final_v3']['mIoU_pct'],57.56699854,places=6)
    def test_instance_layers_and_export_are_real_pngs(self):
        self.assertEqual(self.client.get('/api/jobs/not-valid/layer?instance_id=1').status_code,400)
        self.assertEqual(self.client.get('/api/jobs/'+'a'*24+'/files/secret').status_code,404)
    def test_current_image_scoring_and_export_include_verified_metrics(self):
        # Scorer fixture lives only in this temporary test store. It is never
        # shown in the application or presented as a model accuracy result.
        im=self.image();jid='b'*24;folder=self.app.state.store.jobs/jid;folder.mkdir()
        pred=np.full((64,128),255,np.uint8);pred[:,:64]=0
        Image.fromarray(pred).save(folder/'classes.png')
        (folder/'result.json').write_text(json.dumps({'classes':[{'name':'car'}],
            'files':['classes.png','result.json'],'accuracy':None}),encoding='utf-8')
        with self.app.state.store.db() as c:c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (jid,im['id'],'street','car','["car"]','text','succeeded','complete','test fixture',1,time.time(),None,None,None,1))
        truth=np.zeros((64,128),np.uint8);truth[:,:96]=255;out=io.BytesIO();Image.fromarray(truth).save(out,format='PNG')
        r=self.client.post('/api/jobs/'+jid+'/evaluate',files={'file':('gt.png',out.getvalue(),'image/png')},data={'encoding':'binary'})
        self.assertEqual(r.status_code,200);self.assertAlmostEqual(r.json()['mIoU_pct'],200/3)
        archive=self.client.get('/api/jobs/'+jid+'/files/results.zip')
        self.assertEqual(archive.status_code,200)
        with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
            self.assertIn('accuracy.json',z.namelist());report=json.loads(z.read('result.json'))
            self.assertAlmostEqual(report['accuracy']['mIoU_pct'],200/3)

if __name__=='__main__':unittest.main()
