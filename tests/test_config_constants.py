"""Catch duplicated scientific literals, not arithmetic identities or indices."""
import ast
from pathlib import Path


def test_no_duplicated_scientific_constants():
    # Integer 0/1 (bounds), small integer indices and runtime sizes are structural.
    # Wavelengths and nontrivial numeric science thresholds must come from YAML.
    forbidden = {0.02, 0.30, 47.308, 0.06, 440, 675, 550, 1020, 412, 470, 660,
                 0.2, 0.0001, 0.9, 2010, 0.4, 0.6, 0.7, 0.051, 0.336, 0.085}
    root = Path(__file__).resolve().parents[1]/'dbdust'
    found = []
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value in forbidden:
                found.append(f'{path.name}:{node.lineno}: {node.value}')
    assert not found, 'Scientific values must come from config: '+', '.join(found)
