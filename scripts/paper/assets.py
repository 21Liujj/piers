"""Download only release-pinned, SHA256-verified inference assets."""
from pathlib import Path
import hashlib,json,os,urllib.request,zipfile

REPO=Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def download(asset,path):
    if path.is_file() and path.stat().st_size==asset['bytes'] and sha(path)==asset['sha256']:
        print('VERIFIED',path.name,flush=True);return
    partial=path.with_name(path.name+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>=asset['bytes']:partial.unlink();offset=0
    headers={'User-Agent':'piers-paper-reproduction'}
    if offset:headers['Range']='bytes='+str(offset)+'-'
    with urllib.request.urlopen(urllib.request.Request(asset['url'],headers=headers),timeout=180) as response:
        resume=offset>0 and response.status==206
        if resume and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
            raise ValueError('Unexpected download range')
        with partial.open('ab' if resume else 'wb') as f:
            for block in iter(lambda:response.read(4<<20),b''):f.write(block)
    if partial.stat().st_size!=asset['bytes'] or sha(partial)!=asset['sha256']:
        raise ValueError('Asset checksum failed: '+path.name)
    partial.replace(path);print('DOWNLOADED_AND_VERIFIED',path.name,flush=True)

def fetch(weights_root=None,only=None):
    manifest=json.loads((REPO/'assets/paper20261109/weights.json').read_text())
    root=Path(weights_root or os.environ.get('PIERS_WEIGHTS_ROOT',str(REPO/'assets/weights'))).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    names={a['name'] for a in manifest['assets']}
    if only and set(only)-names:raise ValueError('Unknown assets: '+str(set(only)-names))
    for asset in manifest['assets']:
        if only and asset['name'] not in only:continue
        path=root/asset['name'];download(asset,path)
        if path.suffix!='.zip':continue
        expected={'grounding-dino-tiny-hf/'+n for n in manifest['grounding_dino_files']}
        with zipfile.ZipFile(path) as z:
            if set(z.namelist())!=expected:raise ValueError('Unexpected archive member')
            for member in z.infolist():
                target=(root/member.filename).resolve()
                if not target.is_relative_to(root):raise ValueError('Archive escapes weight root')
                target.parent.mkdir(parents=True,exist_ok=True)
                digest=manifest['grounding_dino_files'][target.name]['sha256']
                if target.exists() and sha(target)==digest:continue
                tmp=target.with_name(target.name+'.partial')
                with z.open(member) as source,tmp.open('wb') as dest:
                    for block in iter(lambda:source.read(4<<20),b''):dest.write(block)
                if sha(tmp)!=digest:raise ValueError('Extracted file checksum failed')
                tmp.replace(target)
    print('ASSETS_READY',root,flush=True)
