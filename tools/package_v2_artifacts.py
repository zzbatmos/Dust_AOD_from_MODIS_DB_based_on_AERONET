"""Copy already fitted inference assets; never fit, select or publish a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--branches', nargs='+', choices=['full8', 'noSSA5', 'minimal2'],
                   default=['full8', 'noSSA5'])
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for branch in args.branches:
        src = args.source / branch
        dst = args.destination / branch
        dst.mkdir(exist_ok=True)
        recorded = json.loads((src/'complete.json').read_text())
        for name in ('classifier.ubj', 'positive_fraction.ubj', 'direct_fraction.ubj',
                     'pair.json', 'postprocessor.npz', 'smear_curve.npz'):
            if sha(src/name) != recorded[name]:
                raise ValueError(f'Source checksum failed: {branch}/{name}')
            shutil.copyfile(src/name, dst/name)
            manifest.append((sha(dst/name), f'{branch}/{name}'))
        metadata = json.loads((src/'metadata.json').read_text())
        public = {key: metadata[key] for key in ('reference', 'branch', 'feature_names',
                   'selected', 'config', 'training_rows', 'training_row_sha256',
                   'source_rows_sha256')}
        public['source_metadata_sha256'] = sha(src/'metadata.json')
        public['source_complete_sha256'] = sha(src/'complete.json')
        (dst/'metadata.json').write_text(json.dumps(public, indent=2)+'\n')
        manifest.append((sha(dst/'metadata.json'), f'{branch}/metadata.json'))
    (args.destination/'MANIFEST.sha256').write_text(''.join(f'{h}  {n}\n' for h,n in sorted(manifest)))
    print('Copied verified inference artifacts; no evaluation results or input data included.')


if __name__ == '__main__':
    main()
