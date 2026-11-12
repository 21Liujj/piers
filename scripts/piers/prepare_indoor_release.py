"""Install the verified indoor checkpoint, calibration and manifests for web use.

Existing files must already match; this never replaces a different experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def install(target, content):
    if target.exists():
        if target.read_bytes() != content:
            raise RuntimeError('Existing file differs; use a separate checkout: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path, help='Existing shared Stage1 checkpoint')
    group.add_argument('--download', action='store_true', help='Download the matching GitHub Release checkpoint')
    args = parser.parse_args()
    assets = json.loads((REPO / 'assets/indoor20261112/release_assets.json').read_text())
    asset = next(a for a in assets['assets'] if a['name'] == 'indoor_stage1_shared.pth')
    old = REPO / 'runs/INDOOR-FOLD1-20261109'
    checkpoint = args.checkpoint.resolve() if args.checkpoint else old / 'train/indoor_lora_last.pth'
    if args.download and not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix('.download')
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'PIERS-indoor-release'})
        print('Downloading verified Stage1 checkpoint...', flush=True)
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as target:
            while block := source.read(1 << 20):
                target.write(block)
        if temporary.stat().st_size != asset['bytes'] or sha(temporary) != asset['sha256']:
            temporary.unlink()
            raise RuntimeError('Downloaded checkpoint failed size/SHA256 verification')
        temporary.rename(checkpoint)
    if checkpoint.stat().st_size != asset['bytes'] or sha(checkpoint) != asset['sha256']:
        raise RuntimeError('Checkpoint failed size/SHA256 verification')
    source = REPO / 'results/indoor_matched20261112'
    protocol = json.loads((source / 'indoor_protocol.json').read_text())
    protocol['shared_checkpoint'] = str(checkpoint)
    root = REPO / 'runs/INDOOR-MATCHED-DINO-20261112'
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Existing protocol differs; use a separate checkout: ' + str(root))
    else:
        install(protocol_path, (json.dumps(protocol, indent=2) + '\n').encode())
    calibration = json.loads((source / 'calibrators/DONE.json').read_text())
    for name in ('teacher_only', 'text_only'):
        model = source / 'calibrators' / (name + '.npz')
        if sha(model) != calibration[name + '_sha256']:
            raise RuntimeError('Calibration fingerprint mismatch: ' + name)
        install(root / 'calibrators' / model.name, model.read_bytes())
    for name in ['split_fit.jsonl', 'split_calib.jsonl', 'split_val.jsonl', 'split_heldout.jsonl', 'SPAN8_CLASS_STATS.json']:
        install(old / name, (source / 'training' / name).read_bytes())
    print('Verified checkpoint and installed indoor web protocol/calibrators/manifests.')
    print('Indoor checkout:', REPO)
    print('Grounding DINO/SAM2 weights are still required under /root/autodl-tmp/weights.')


if __name__ == '__main__':
    main()
