import pytest
import subprocess
import sys
from pathlib import Path
import os
import re

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
example_scripts = [p for p in EXAMPLES_DIR.glob('*.py')]

@pytest.mark.parametrize("script_path", example_scripts, ids=lambda p: p.name)
def test_example_runs_without_error(script_path):
    with open(script_path, 'r') as f:
        original_code = f.read()
    
    modified_code = re.sub(r"\.render\(.*\)", ".to_glsl()", original_code)

    try:
        env = os.environ.copy()
        project_root = str(Path(__file__).parent.parent)
        env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            [sys.executable, '-c', modified_code],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            check=True
        )
    
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script {script_path.name} failed with exit code {e.returncode}.\n"
                    f"STDOUT:\n{e.stdout}\n"
                    f"STDERR:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {script_path.name} timed out.")