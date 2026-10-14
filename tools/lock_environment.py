"""Record the installed dependency closure without accessing any credentials."""
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    root = Path(__file__).resolve().parents[1]
    pending = ['numpy', 'pandas', 'scipy', 'xgboost', 'scikit-learn', 'xarray',
               'netCDF4', 'pyhdf', 'PyYAML', 'threadpoolctl', 'pyarrow', 'pytest',
               'setuptools', 'wheel', 'packaging']
    locked = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in locked:
            continue
        dist = metadata.distribution(name)
        locked[name] = dist.version
        for text in dist.requires or []:
            dep = Requirement(text)
            if dep.marker is None or dep.marker.evaluate({'extra': ''}):
                pending.append(dep.name)
    (root/'requirements-lock.txt').write_text(
        '# Verified Linux x86_64 / Python '+platform.python_version()+'\n'+
        ''.join(f'{name}=={version}\n' for name,version in sorted(locked.items())))
    (root/'environment.yml').write_text(
        'name: dbdust-v2\nchannels:\n  - conda-forge\ndependencies:\n  - python='+
        platform.python_version()+'\n  - pip\n  - pip:\n      - -r requirements-lock.txt\n')
    (root/'docs/v2/environment.json').write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(), machine=platform.machine(),
        dependencies=locked), indent=2)+'\n')


if __name__ == '__main__':
    main()
