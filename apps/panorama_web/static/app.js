'use strict';
const $=id=>document.getElementById(id);
const S={scene:'street',image:null,job:null,view:'original',scale:1,panX:0,panY:0,focus:null,connected:false,cap:null,experiments:null,poll:null};
const pending=s=>['queued','running','cancelling'].includes(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=2)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);
const seconds=v=>v==null?'—':`${num(v,1)} s`;
const date=v=>v?new Date(v*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}):'—';
const fileUrl=name=>`/api/jobs/${S.job.id}/files/${name}`;
let toastTimer,connectionBusy=false;
async function api(path,options={}){
  const opt={...options};if(opt.body&&!(opt.body instanceof FormData)){opt.headers={'Content-Type':'application/json',...(opt.headers||{})};opt.body=JSON.stringify(opt.body);}
  let r;try{r=await fetch(path,opt);}catch(e){throw new Error('无法连接本机服务，请检查启动窗口。');}
  let data;try{data=await r.json();}catch(e){throw new Error(`服务未返回有效结果（${r.status}）`);}
  if(!r.ok){let msg=data.detail||data.error||`请求失败（${r.status}）`;if(Array.isArray(msg))msg=msg.map(x=>x.msg).join('；');throw new Error(msg);}
  return data;
}
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),5000);}
function toggle(id,show){$(id).classList.toggle('hidden',!show);}
function errorBox(id,message){$(id).textContent=message||'';toggle(id,!!message);}
function showPage(name){
  if(!['workbench','experiments','history'].includes(name))name='workbench';
  document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id==='page-'+name));
  document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x.dataset.page===name));
  $('breadcrumb').textContent={workbench:'分割工作台',experiments:'实验对比',history:'任务记录'}[name];
  if(location.hash!=='#'+name)history.replaceState(null,'','#'+name);
  if(name==='experiments')loadExperiments();if(name==='history')refreshHistory();requestAnimationFrame(fitImage);
}
function sceneChanged(scene){
  S.scene=scene;document.querySelectorAll('[data-scene]').forEach(x=>x.classList.toggle('selected',x.dataset.scene===scene));
  $('sceneNote').classList.toggle('indoor',scene==='indoor');
  $('sceneNote').innerHTML='<span class="tiny-dot"></span>'+ (scene==='street'?'完整模型 · I1 / I2 / I3':'室内适配 · I1 / I2 / I3');
  const cap=S.cap?.scenes.find(x=>x.id===scene),names=cap?.classes||[];
  $('defaultClassCount').textContent=names.length+' 类';
  $('defaultClassList').innerHTML=names.map(n=>`<span title="${esc(n)}">${esc(S.cap?.labels_zh[n]||n)}</span>`).join('');
  const chips=scene==='street'?[['car','汽车'],['person','行人'],['bus','公交车']]:[['chair','椅子'],['table','桌子'],['sofa','沙发']];
  $('queryChips').innerHTML=chips.map(([n,z])=>`<button data-query="${n}" type="button">${z}</button>`).join('');
  $('queryInput').placeholder=scene==='street'?'例如：汽车，行人，公交车\n或 car, person, bus':'例如：椅子，桌子，沙发\n或 chair, table, sofa';
}
async function upload(file){
  if(!file||pending(S.job?.status))return;
  if(file.size>30*1024*1024)return toast('请选择30MB以内的图片。');
  const form=new FormData();form.append('file',file);$('uploadTitle').textContent='正在读取图片…';
  try{const im=await api('/api/images',{method:'POST',body:form});clearTimeout(S.poll);S.job=null;setImage(im);clearResults();setView('original');toast('图片已就绪，可指定目标或直接开始分割。');}
  catch(e){toast(e.message);}finally{$('uploadTitle').textContent='点击上传或拖入图片';$('imageInput').value='';}
}
function setImage(im){
  S.image=im;$('imageName').textContent=im.name;$('imageName').title=im.name;$('imageDimensions').textContent=`${im.width} × ${im.height} px`;
  toggle('uploadZone',false);toggle('imageInfo',true);toggle('viewerEmpty',false);toggle('imageStack',true);toggle('viewLabel',true);
  $('baseImage').src=`/api/images/${im.id}/original`;S.scale=1;S.panX=S.panY=0;fitImage();$('runButton').disabled=false;
}
function clearResults(){S.focus=null;['resultStats','resultPanel','taskError','taskProgress','viewerLoading','clearFocus'].forEach(id=>toggle(id,false));$('downloadButton').disabled=true;document.querySelectorAll('[data-view]').forEach(b=>b.disabled=b.dataset.view!=='original');$('runButton').disabled=!S.image;setInputsDisabled(false);}
function setInputsDisabled(value){$('imageInput').disabled=value;$('queryInput').disabled=value;document.querySelectorAll('[data-scene],[data-query]').forEach(x=>x.disabled=value);$('replaceImage').disabled=value;}
async function startJob(){
  if(!S.image)return toast('请先上传一张图片。');if(pending(S.job?.status))return;
  if(!S.connected){toast('请先连接 GPU 服务器。');openSettings();return;}
  try{clearResults();$('runButton').disabled=true;S.job=await api('/api/jobs',{method:'POST',body:{image_id:S.image.id,scene:S.scene,text:$('queryInput').value}});updateJob(S.job);pollJob(S.job.id);}
  catch(e){$('runButton').disabled=false;toast(e.message);}
}
async function pollJob(id){
  clearTimeout(S.poll);
  try{const row=await api('/api/jobs/'+id);if(S.job?.id!==id)return;updateJob(row);if(pending(row.status))S.poll=setTimeout(()=>pollJob(id),1000);else refreshHistoryCount();}
  catch(e){if(S.job?.id===id){$('progressMessage').textContent='连接暂时中断，正在重新获取进度…';S.poll=setTimeout(()=>pollJob(id),3000);}}
}
function updateJob(row){
  S.job=row;const busy=pending(row.status);setInputsDisabled(busy);$('runButton').disabled=busy||!S.image;
  $('runButton').querySelector('span').textContent=busy?'正在处理':'开始分割';toggle('taskProgress',busy);toggle('viewerLoading',busy);$('cancelButton').disabled=row.status==='cancelling';
  if(busy){
    const title={queued:'等待处理',waiting_gpu:'等待 GPU 空闲',uploading:'发送图像',loading:'加载模型',text:'文本目标分割',semantic:'语义协同推理',rendering:'生成全景结果',downloading:'保存结果',cancelling:'正在取消'};
    $('progressTitle').textContent=title[row.stage]||'处理任务';$('progressMessage').textContent=row.message||'';
    $('progressBar').style.width=Math.max(2,(row.progress||0)*100)+'%';$('progressTime').textContent=seconds(Date.now()/1000-(row.started||row.created));
    $('viewerStage').textContent=row.message||'请稍候';toggle('taskError',false);
  }else if(row.status==='succeeded'&&row.result){renderResult(row);if(S.view==='original')setView('semantic');}
  else if(row.status==='failed'||row.status==='interrupted'){
    $('taskError').innerHTML=`<strong>${esc(row.message||'任务未完成')}</strong>${row.error?`<details><summary>查看错误详情</summary><pre>${esc(row.error)}</pre></details>`:''}<p>检查服务器连接后，可点击“开始分割”重新运行。</p>`;toggle('taskError',true);
  }else if(row.status==='cancelled'){toast('任务已取消。');toggle('taskError',false);}
}
function renderResult(row){
  const r=row.result;toggle('resultStats',true);toggle('resultPanel',true);$('downloadButton').disabled=false;
  document.querySelectorAll('[data-view]').forEach(b=>b.disabled=false);
  $('statTime').textContent=seconds(row.elapsed);const t=r.audit?.timing||{};
  $('statTimeDetail').textContent=t.warm_processing_seconds==null?
    `含等待、加载与传输 · 服务处理 ${seconds(t.server_processing_seconds)}（含加载）`:
    `含等待、加载与传输 · GPU服务 ${seconds(t.warm_processing_seconds)}`;
  $('statClasses').textContent=r.classes.filter(x=>x.pixels>0).length+' / '+r.classes.length;
  $('statClassDetail').textContent=row.mode==='text'?'按指定文本查询':'按场景默认类别分割';$('statInstances').textContent=r.instance_count;
  $('instanceCount').textContent=r.instance_count;$('resultModel').textContent=row.scene==='indoor'?(r.audit?.indoor_adapted?'室内适配完整模型':'室内通用文本'):'完整模型 · v3';
  $('classResults').innerHTML=r.classes.map(c=>`<button class="class-item ${c.pixels?'':'zero'}" data-class="${c.index}" title="${esc(c.name)} · 类别 ID ${c.index}"><span class="swatch" style="background:${c.color}"></span><div><strong>${esc(c.name_zh)}</strong><small>${esc(c.name)} · #${c.index}</small></div><span>${num(c.fraction*100,1)}%</span></button>`).join('');
  $('instanceResults').innerHTML=r.instances.length?r.instances.map(i=>`<button class="instance-item" data-instance="${i.id}"><span class="swatch" style="background:${i.color}"></span><strong>${esc(i.label)}</strong><small>检测 ${num(i.detector_score,2)} · SAM ${num(i.sam_quality,2)} · ${i.views.length} 视窗</small></button>`).join('')+'<p class="field-hint">实例数量为模型估计，可能存在漏检或重复；检测分数与 SAM 质量不代表准确率。</p>':'<div class="empty-inline">未检出独立实例。<br>类别区域仍可在“语义分割”中查看。</div>';
  $('accuracyHint').textContent='无标准标签，未计算准确率';toggle('accuracyResult',false);
  if(row.accuracy)renderAccuracy(row.accuracy);$('auditContent').textContent=JSON.stringify(r.audit,null,2);renderViewer();
}
function setView(view){S.view=view;S.focus=null;renderViewer();}
function renderViewer(){
  document.querySelectorAll('[data-view]').forEach(x=>{x.classList.toggle('selected',x.dataset.view===S.view);x.setAttribute('aria-selected',x.dataset.view===S.view);});
  const has=!!S.job?.result&&S.job.status==='succeeded';let mask=null;
  if(S.focus&&has)mask=`/api/jobs/${S.job.id}/layer?${S.focus.type}=${S.focus.id}`;
  else if(has&&S.view==='semantic')mask=fileUrl('semantic.png');
  else if(has&&S.view==='instances')mask=fileUrl('instance_overlay.jpg');
  toggle('maskImage',!!mask);if(mask)$('maskImage').src=mask;
  $('maskImage').style.opacity=S.view==='instances'&&!S.focus?'1':String(Number($('opacitySlider').value)/100);
  const compare=has&&S.view==='compare'&&!S.focus;toggle('compareImage',compare);toggle('compareLine',compare);toggle('compareControl',compare);
  if(compare){$('compareImage').src=fileUrl('overlay.jpg');updateCompare();}
  toggle('opacityControl',has&&(S.view==='semantic'||!!S.focus));toggle('clearFocus',!!S.focus);
  $('viewLabel').textContent=S.focus?S.focus.label:{original:'原始图像',semantic:'语义类别',instances:'独立实例',compare:'左侧原图 / 右侧分割结果'}[S.view];
  $('viewerHint').textContent=S.image?'拖动画布查看 · 用 + / − 调整缩放':'支持室外街景与室内全景图像';
}
function updateCompare(){const x=$('compareSlider').value;$('compareImage').style.clipPath=`inset(0 0 0 ${x}%)`;$('compareLine').style.left=x+'%';}
function fitImage(){
  if(!S.image)return;const box=$('viewport').getBoundingClientRect();if(!box.width)return;
  const ratio=Math.min((box.width-24)/S.image.width,(box.height-30)/S.image.height);$('imageStack').style.width=S.image.width*ratio+'px';$('imageStack').style.height=S.image.height*ratio+'px';applyTransform();
}
function applyTransform(){$('imageStack').style.transform=`translate(calc(-50% + ${S.panX}px),calc(-50% + ${S.panY}px)) scale(${S.scale})`;$('zoomValue').textContent=Math.round(S.scale*100)+'%';}
function zoom(delta){if(!S.image)return;S.scale=Math.min(5,Math.max(1,S.scale+delta));if(S.scale===1)S.panX=S.panY=0;applyTransform();}
function focusMask(type,id){if(!S.job?.result)return;const r=S.job.result;const item=type==='class_index'?r.classes.find(x=>x.index===id):r.instances.find(x=>x.id===id);S.focus={type,id,label:item?.label||item?.name_zh||item?.name||'目标'};S.view='semantic';renderViewer();}
async function loadJob(id){
  try{const row=await api('/api/jobs/'+id);clearTimeout(S.poll);setImage({id:row.image_id,name:row.image_name,width:row.width,height:row.height});clearResults();sceneChanged(row.scene);$('queryInput').value=row.text;S.view='original';updateJob(row);showPage('workbench');if(pending(row.status))pollJob(id);window.scrollTo({top:0,behavior:'smooth'});}
  catch(e){toast(e.message);}
}
async function refreshHistoryCount(){try{const {jobs}=await api('/api/jobs');$('historyCount').textContent=jobs.length||'';}catch(e){}}
async function refreshHistory(){
  try{const {jobs}=await api('/api/jobs');$('historyCount').textContent=jobs.length||'';
    $('historyList').innerHTML=jobs.length?jobs.map(j=>`<article class="history-card"><img src="/api/images/${j.image_id}/preview" alt="${esc(j.image_name)}" loading="lazy"><div><h3>${esc(j.image_name)}</h3><p>${j.scene==='street'?'室外街景':'室内场景'} · ${j.mode==='text'?esc(j.classes.join(', ')):'默认类别全图分割'}</p><small>${date(j.created)}${j.elapsed?' · '+seconds(j.elapsed):''}</small></div><span class="task-status ${j.status}">${{succeeded:'已完成',failed:'失败',running:'处理中',queued:'排队中',cancelling:'取消中',cancelled:'已取消',interrupted:'已中断'}[j.status]||esc(j.status)}</span><button class="small-button" data-job="${j.id}">${j.status==='succeeded'?'查看结果':'查看任务'} <span>→</span></button></article>`).join(''):'<div class="history-empty"><svg><use href="#i-clock"/></svg>还没有任务记录。上传一张全景图，开始第一次分割。</div>';
  }catch(e){toast(e.message);}
}
async function refreshConnection(){
  if(connectionBusy)return;connectionBusy=true;
  try{const data=await api('/api/connection');const h=data.health;S.connected=!!h.connected;$('connectionBadge').classList.toggle('connected',S.connected);$('connectionBadge').classList.toggle('disconnected',!S.connected);
    $('connectionText').textContent=S.connected?(h.other_gpu_processes?'GPU 实验进行中':h.active_job?'GPU 正在处理':'GPU 服务器已连接'):'GPU 尚未连接';$('connectionBadge').title=h.other_gpu_processes?'网页推理会等待其他GPU实验结束':h.message||'';
    return data;
  }catch(e){S.connected=false;$('connectionText').textContent='本机服务未连接';}finally{connectionBusy=false;}
}
async function openSettings(){
  errorBox('connectionError','');$('serverPassword').value='';if(!$('settingsDialog').open)$('settingsDialog').showModal();
  try{const data=await api('/api/connection');const c=data.config;for(const [id,key] of [['serverHost','host'],['serverPort','port'],['serverUser','username'],['serverRepo','repo'],['serverPython','python'],['serverWeights','weights'],['serverRuntime','runtime']])$(id).value=c[key]||'';
    const h=data.health;$('connectionDetails').textContent=h.connected?`已连接 · ${h.models?.length?'模型已加载':'模型按需加载'} · 空闲 ${h.idle_release_seconds||90} 秒自动释放GPU。${h.active_job?' 当前有任务运行。':''}`:h.message||'尚未连接';
  }catch(e){errorBox('connectionError',e.message);}
}
async function connectSubmit(e){
  e.preventDefault();errorBox('connectionError','');$('connectButton').disabled=true;$('connectButton').textContent='正在连接…';
  const body={host:$('serverHost').value.trim(),port:Number($('serverPort').value),username:$('serverUser').value.trim(),password:$('serverPassword').value};
  for(const [id,key] of [['serverRepo','repo'],['serverPython','python'],['serverWeights','weights'],['serverRuntime','runtime']])if($(id).value.trim())body[key]=$(id).value.trim();
  $('serverPassword').value='';
  try{await api('/api/connection',{method:'POST',body});await refreshConnection();$('settingsDialog').close();toast('GPU 服务器已连接，可以开始分割。');}
  catch(err){errorBox('connectionError',err.message);}finally{$('connectButton').disabled=false;$('connectButton').textContent='连接并检查';body.password='';}
}
async function loadExperiments(){
  if(S.experiments){renderExperiments();return;}
  try{S.experiments=await api('/api/experiments');$('datasetSelect').innerHTML=S.experiments.rows.map((r,i)=>`<option value="${i}">${esc(r.evaluation_space)}</option>`).join('');renderExperiments();}
  catch(e){toast(e.message);}
}
function renderExperiments(){
  if(!S.experiments)return;const metric=$('metricSelect').value;
  $('experimentRows').innerHTML=S.experiments.rows.map(r=>{const a=r.original[metric],b=r.full_final_v3[metric],delta=b-a,ci=metric==='mIoU_pct'?r.delta_mIoU_pp_95CI:null;return `<tr><td>${esc(r.evaluation_space)}</td><td>${r.images}</td><td>${num(a)}%</td><td class="metric-full">${num(b)}%</td><td class="${delta>=0?'delta-positive':'delta-negative'}">${delta>=0?'+':''}${num(delta)}${ci?`<span class="ci-note">95% CI [${num(ci[0])}, ${num(ci[1])}]</span>`:''}</td></tr>`;}).join('');
  renderClassChart();const pairs={};for(const r of S.experiments.efficiency){(pairs[r.scenario]||={})[r.method]=r;}
  const names={DensePASS_shared13:'DensePASS · 13类',WildPASS_native8:'WildPASS · 8类',SynPASS_native22:'SynPASS · 22类',SynPASS_shared13:'SynPASS · 13类'};
  $('efficiencyRows').innerHTML=Object.entries(pairs).map(([name,p])=>`<div class="efficiency-item"><h3>${esc(names[name]||name)}</h3><p><span>原始</span><strong>${seconds(p.original?.mean_seconds)}</strong></p><p><span>完整</span><strong>${seconds(p.full?.mean_seconds)}</strong></p><p><span>设备峰值显存</span><span>${num((p.original?.peak_device_used_mib_sampled||0)/1024)} / ${num((p.full?.peak_device_used_mib_sampled||0)/1024)} GiB</span></p></div>`).join('');
}
function renderClassChart(){
  if(!S.experiments)return;const r=S.experiments.rows[Number($('datasetSelect').value)||0];
  $('classChart').innerHTML=r.classes.map((c,i)=>{const a=r.original.per_class_IoU_pct[i],b=r.full_final_v3.per_class_IoU_pct[i];return `<div class="bar-row"><span class="bar-name" title="${esc(c)}">${esc(S.cap?.labels_zh[c]||c)}</span><div class="bar-tracks"><div class="bar-track"><span class="baseline" style="width:${Math.max(0,a||0)}%"></span><small>${num(a,1)}</small></div><div class="bar-track"><span class="full" style="width:${Math.max(0,b||0)}%"></span><small>${num(b,1)}</small></div></div></div>`;}).join('');
}
function renderAccuracy(a){$('accuracyHint').textContent='已使用标准标签评估当前图像';const pct=x=>x==null?'未定义':num(x)+'%';$('accuracyResult').innerHTML=`<span><small>单图 mIoU</small>${pct(a.mIoU_pct)}</span><span><small>mDice</small>${pct(a.mDice_pct)}</span><span><small>Pixel Acc.</small>${pct(a.pixel_accuracy_pct)}</span>`;toggle('accuracyResult',true);}
async function scoreSubmit(e){
  e.preventDefault();errorBox('gtError','');if(!S.job?.result)return;
  const form=new FormData();form.append('file',$('gtFile').files[0]);form.append('encoding',$('gtEncoding').value);$('scoreButton').disabled=true;
  try{const a=await api(`/api/jobs/${S.job.id}/evaluate`,{method:'POST',body:form});S.job.accuracy=a;renderAccuracy(a);$('evaluateDialog').close();toast('已计算当前图像指标。');}
  catch(err){errorBox('gtError',err.message);}finally{$('scoreButton').disabled=false;}
}

document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>showPage(b.dataset.page)));
document.querySelector('.brand').addEventListener('click',()=>showPage('workbench'));
window.addEventListener('hashchange',()=>showPage(location.hash.slice(1)));
$('imageInput').addEventListener('change',()=>upload($('imageInput').files[0]));
for(const id of ['replaceImage','emptyUploadButton'])$(id).addEventListener('click',()=>$('imageInput').click());
$('uploadZone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('imageInput').click();}});
for(const kind of ['dragenter','dragover'])$('uploadZone').addEventListener(kind,e=>{e.preventDefault();$('uploadZone').classList.add('dragging');});
for(const kind of ['dragleave','drop'])$('uploadZone').addEventListener(kind,e=>{e.preventDefault();$('uploadZone').classList.remove('dragging');if(kind==='drop')upload(e.dataTransfer.files[0]);});
document.querySelectorAll('[data-scene]').forEach(b=>b.addEventListener('click',()=>sceneChanged(b.dataset.scene)));
$('queryChips').addEventListener('click',e=>{const b=e.target.closest('[data-query]');if(b&&!b.disabled){const v=$('queryInput').value.trim();$('queryInput').value=v?v+', '+b.dataset.query:b.dataset.query;$('queryInput').focus();}});
$('runButton').addEventListener('click',startJob);document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!document.querySelector('dialog[open]')){e.preventDefault();startJob();}});
$('cancelButton').addEventListener('click',async()=>{if(!S.job)return;try{updateJob(await api('/api/jobs/'+S.job.id+'/cancel',{method:'POST',body:{}}));}catch(e){toast(e.message);}});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
$('downloadButton').addEventListener('click',()=>{if(S.job?.result)location.href=fileUrl('results.zip');});
$('compareSlider').addEventListener('input',updateCompare);$('opacitySlider').addEventListener('input',renderViewer);
$('zoomIn').addEventListener('click',()=>zoom(.25));$('zoomOut').addEventListener('click',()=>zoom(-.25));$('zoomFit').addEventListener('click',()=>{S.scale=1;S.panX=S.panY=0;fitImage();});
let drag=null;$('viewport').addEventListener('pointerdown',e=>{if(S.image&&S.scale>1&&!e.target.closest('button,input')){drag={x:e.clientX,y:e.clientY,px:S.panX,py:S.panY};$('viewport').setPointerCapture(e.pointerId);$('imageStack').classList.add('dragging');}});
$('viewport').addEventListener('pointermove',e=>{if(drag){S.panX=drag.px+e.clientX-drag.x;S.panY=drag.py+e.clientY-drag.y;applyTransform();}});
for(const ev of ['pointerup','pointercancel'])$('viewport').addEventListener(ev,()=>{drag=null;$('imageStack').classList.remove('dragging');});
$('viewport').addEventListener('wheel',e=>{if(e.ctrlKey&&S.image){e.preventDefault();zoom(e.deltaY<0?.1:-.1);}},{passive:false});
new ResizeObserver(fitImage).observe($('viewport'));$('baseImage').addEventListener('load',fitImage);
document.querySelectorAll('[data-result-tab]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-result-tab]').forEach(x=>x.classList.toggle('selected',x===b));toggle('classResults',b.dataset.resultTab==='classes');toggle('instanceResults',b.dataset.resultTab==='instances');}));
$('classResults').addEventListener('click',e=>{const x=e.target.closest('[data-class]');if(x)focusMask('class_index',Number(x.dataset.class));});
$('instanceResults').addEventListener('click',e=>{const x=e.target.closest('[data-instance]');if(x)focusMask('instance_id',Number(x.dataset.instance));});
$('clearFocus').addEventListener('click',()=>{S.focus=null;renderViewer();});
$('settingsButton').addEventListener('click',openSettings);$('connectionBadge').addEventListener('click',openSettings);$('connectionForm').addEventListener('submit',connectSubmit);
document.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>$(b.dataset.close).close()));
$('unloadButton').addEventListener('click',async()=>{try{await api('/api/models/unload',{method:'POST',body:{}});toast('已请求释放GPU模型；下次任务将重新加载。');setTimeout(openSettings,1800);}catch(e){errorBox('connectionError',e.message);}});
$('refreshHistory').addEventListener('click',refreshHistory);$('historyList').addEventListener('click',e=>{const b=e.target.closest('[data-job]');if(b)loadJob(b.dataset.job);});
$('metricSelect').addEventListener('change',renderExperiments);$('datasetSelect').addEventListener('change',renderClassChart);
$('auditButton').addEventListener('click',()=>$('auditDialog').showModal());
$('evaluateButton').addEventListener('click',()=>{if(!S.job?.result)return;errorBox('gtError','');$('gtFile').value='';const single=S.job.classes.length===1;$('gtEncoding').querySelector('[value=binary]').disabled=!single;$('gtEncoding').value=single?'binary':'class_ids';$('evaluateDialog').showModal();});
$('evaluateForm').addEventListener('submit',scoreSubmit);
(async()=>{try{S.cap=await api('/api/capabilities');sceneChanged('street');}catch(e){toast(e.message);}showPage(location.hash.slice(1)||'workbench');refreshConnection();refreshHistoryCount();setInterval(refreshConnection,10000);})();
