from pathlib import Path
from dataclasses import dataclass, field
import platform
import subprocess
import xml.etree.ElementTree as ET
import json
import ast
from ast import AsyncFunctionDef, ClassDef, FunctionDef

from .snippets import _node_to_obj

REPORT = Path("./src/profiler/temp/report.xml")

PROFILES_DIR = Path("./src/profiler/profiles")
VENVS_DIR = Path("./repos/venvs")
PROJECTS_DIR = Path("./repos/projects")

NODE_TYPES = (FunctionDef, AsyncFunctionDef, ClassDef)

def _venv_python_path(proj_name: str) -> Path:
    if platform.system() == "Windows":
        return VENVS_DIR / proj_name / "Scripts" / "python.exe"
    return VENVS_DIR / proj_name / "bin" / "python"


@dataclass
class ProjProfile:
    proj_name: str

    baseline_runs: int = 10
    top_k: int = 10
    
    top_bottlenecks: list = field(default_factory=list)
    start_runtime: float | None = None
    failure_count: int | None = None
    
    venv_python: Path | None = None
    repo_path: Path | None = None
    report_path: Path | None = None
    raw_profile_path: Path | None = None
    filtered_profile_path: Path | None = None

    def __post_init__(self):
        self.repo_path = (PROJECTS_DIR / self.proj_name).resolve()
        self.report_path = REPORT.resolve()
        self.raw_profile_path = (PROFILES_DIR / f"{self.proj_name}_raw.speedscope").resolve()
        self.filtered_profile_path = (PROFILES_DIR / f"{self.proj_name}_filtered.speedscope").resolve()
        self.venv_python = _venv_python_path(self.proj_name).resolve()

    def initialize(self, setup=False):
        running_total = 0.0
        last_failure_count = None
        profile_count = self.baseline_runs if not setup else 1
        for _ in range(profile_count):
            last_failure_count, duration = _construct_profile(
                proj_name=self.proj_name,
                venv_python=self.venv_python,
                output_file=self.raw_profile_path,
                filtered_output_file=self.filtered_profile_path,
                repo_path=self.repo_path,
                report_path=self.report_path,
            )
            running_total += duration
        self.start_runtime = running_total / (profile_count)

        self.failure_count = last_failure_count
        if self.failure_count is None:
            raise RuntimeError(f"No failure count available for project {self.proj_name}")
        self.top_bottlenecks = _speedscope_bottlenecks(self.filtered_profile_path, self.top_k)

    def yield_snippet(self):
        for node in self.top_bottlenecks:
            yield _node_to_obj(node, self.repo_path)

    def check_patch(self):
        new_failure_count, new_runtime = _construct_profile(
            proj_name=self.proj_name,
            venv_python=self.venv_python,
            output_file=self.raw_profile_path,
            filtered_output_file=self.filtered_profile_path,
            repo_path=self.repo_path,
            report_path=self.report_path,
        )

        if new_failure_count is None:
            raise RuntimeError(f"No failure count available for project {self.proj_name}")
        if self.failure_count is None:
            raise RuntimeError("Baseline failure count is not initialized")
        if new_failure_count > self.failure_count:
            raise RuntimeError(
                f"Patch increased failures for {self.proj_name}: {self.failure_count} -> {new_failure_count}"
            )

        self.failure_count = new_failure_count
        return new_runtime
    
    def new_average_runtime(self):
        running_total = 0.0
        for _ in range(10):
            duration = self.check_patch()
            running_total += duration
        return running_total / 10

    
def _construct_profile(
    proj_name: str,
    venv_python: Path,
    output_file: Path,
    filtered_output_file: Path,
    repo_path: Path,
    report_path: Path,
):
    
    print("Running py-spy profiler...")
    try:
        subprocess.run(
            [
                "py-spy",
                "record",
                "-f",
                "speedscope",
                "--full-filenames",
                "-o",
                str(output_file),
                "--subprocesses",
                "--",
                str(venv_python),
                "-m",
                "pytest",
                str(repo_path),
                "--tb=short",
                f"--junit-xml={report_path}",
            ],
            capture_output=False,
            cwd=repo_path,
        )
    except KeyboardInterrupt:
        print("Tests halted - speedscope saved")

    root = ET.parse(report_path).getroot()
    report = root if root.tag == 'testsuite' else root.find('testsuite')
    errors = int(report.get('errors', 0))
    if errors > 0:
        raise RuntimeError(f"Tests had {errors} errors - cannot proceed with profiling")
    
    failure_count = int(report.get('failures', 0))
    duration = float(report.get('time', 0.0))

    # finally generate filtered speedscope
    _filter_speedscope(
        proj_name=proj_name,
        input_file=output_file,
        output_file=filtered_output_file,
        project_path=repo_path,
    )
    return failure_count, duration


def _filter_speedscope(proj_name: str, input_file: Path, output_file: Path, project_path: Path):
    """
    Filter speedscope profile to keep only functions from a given project.
    Removes external library calls and import statements.
    """
    project_abs = str(project_path).replace('\\', '/')
        
    # Load the speedscope file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get frames from shared section
    shared = data.get('shared', {})
    if 'frames' not in shared:
        print("ERROR: No frames found in shared section")
        return
    
    original_frames = shared['frames']
        
    # Filter frames
    filtered_frames = []
    frame_index_map = {}  # Map old index to new index
    
    for idx, frame in enumerate(original_frames):
        frame_name = frame.get('name', '')
        frame_file = frame.get('file', '')
        
        # Normalize file path for comparison
        if frame_file:
            frame_file_normalized = frame_file.replace('\\', '/')
        else:
            frame_file_normalized = ''
        
        # Skip import statements and frozen importlib
        if '<frozen importlib' in frame_file or 'import>' in frame_name:
            continue
        
        # Skip <module> frames that are on import lines
        if frame_name == '<module>' and frame_file and _is_import_line(frame_file, frame.get('line', 0)):
            continue
        
        # Skip test files
        if '/tests/' in frame_file_normalized or '\\tests\\' in frame_file_normalized:
            continue
        
        # Skip frames that mention tests in their name (like pytest commands)
        if 'pytest' in frame_name.lower() or '/tests' in frame_name or '\\tests' in frame_name:
            continue
        
        # Keep frames from  project
        if frame_file_normalized and project_abs.lower() in frame_file_normalized.lower():
            frame_index_map[idx] = len(filtered_frames)
            filtered_frames.append(frame)
        # Also keep frames without file info but with project-related names (but not test-related)
        elif not frame_file_normalized and (proj_name in frame_name.lower()):
            frame_index_map[idx] = len(filtered_frames)
            filtered_frames.append(frame)
        
    # Update the shared frames
    data['shared']['frames'] = filtered_frames
    
    # Update sample indices in all profiles to match new frame indices
    for profile in data.get('profiles', []):
        if 'samples' in profile:
            new_samples = []
            new_weights = []
            original_weights = profile.get('weights', [])
            
            for i, sample in enumerate(profile['samples']):
                if isinstance(sample, list):
                    # Remap frame indices in the stack
                    new_sample = [frame_index_map[frame_idx] for frame_idx in sample if frame_idx in frame_index_map]
                    if new_sample:  # Only keep non-empty samples
                        new_samples.append(new_sample)
                        # Keep corresponding weight if it exists
                        if i < len(original_weights):
                            new_weights.append(original_weights[i])
                else:
                    if sample in frame_index_map:
                        new_samples.append(frame_index_map[sample])
                        # Keep corresponding weight if it exists
                        if i < len(original_weights):
                            new_weights.append(original_weights[i])
            
            profile['samples'] = new_samples
            if original_weights:
                profile['weights'] = new_weights
    
    # Save the filtered profile
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    

def _speedscope_bottlenecks(filtered_file: Path, top_k: int = 10):
    if not filtered_file.exists():
        raise FileNotFoundError(f"Filtered profile not found: {filtered_file}")
    
    # Load the filtered speedscope file
    with open(filtered_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get frames from shared section
    shared = data.get('shared', {})
    frames = shared.get('frames', [])
    
    if not frames:
        print("ERROR: No frames found in filtered profile")
        return []
    
    # frame time tracking
    frame_times = {}  # {frame_index : total_time}
    for profile in data.get('profiles', []):
        samples, weights = profile.get('samples', []), profile.get('weights', [])
    
        # weights rep the time on each sample
        for i, sample in enumerate(samples):
            weight = weights[i] if i < len(weights) else 1
    
            # add weight to each frame in the stack
            if isinstance(sample, list):
                for frame_idx in sample:
                    if frame_idx not in frame_times:
                        frame_times[frame_idx] = 0
                    frame_times[frame_idx] += weight
            else:
                if sample not in frame_times:
                    frame_times[sample] = 0
                frame_times[sample] += weight
    
    # sort frames by total time (descending)
    sorted_frames = sorted(frame_times.items(), key=lambda x: x[1], reverse=True)
    seen_names = set()
    seen_nodes = set()
    
    candidates = []

    for frame_idx, _ in sorted_frames:
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            frame_name = frame.get('name', '')
            # avoid dupes
            if frame_name not in seen_names:
                seen_names.add(frame_name)
                file_path = frame.get('file', '')
                line_no = frame.get('line', 0)

                if not file_path or line_no <= 0:
                    continue

                node = _get_node(file_path, line_no)
                if node is None:
                    continue

                node_dump = ast.dump(node)
                if node_dump not in seen_nodes:
                    seen_nodes.add(node_dump)
                    candidates.append(node)

    # enforce exclusive scopes: drop nodes nested inside another candidate
    top_nodes = _filter_exclusive_scopes(candidates, top_k)
    
    if len(top_nodes) < top_k:
        print("WARNING: Not enough top nodes found in profile")

    return top_nodes


def _filter_exclusive_scopes(candidates, top_k: int):
    """Remove candidates whose range is contained within another candidate."""
    def _node_range(node):
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start = node.decorator_list[0].lineno
        else:
            start = node.lineno
        return node.filename, start, node.end_lineno

    ranges = [_node_range(n) for n in candidates]
    result = []
    for i, (n, (f_i, s_i, e_i)) in enumerate(zip(candidates, ranges)):
        nested = False
        for j, (f_j, s_j, e_j) in enumerate(ranges):
            if i != j and f_i == f_j and s_j <= s_i and e_i <= e_j:
                nested = True
                break
        if not nested:
            result.append(n)
            if len(result) >= top_k:
                break
    return result

def _is_import_line(file_path: str, line_number: int) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
                        
        line = lines[line_number - 1].strip()
        
        if line.startswith('import ') or line.startswith('from '):
            return True
            
        return False
    except Exception:
        return False


def _get_node(abs_path : str, target : int):
    if target <= 0:
        return None

    with open(abs_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    tree = ast.parse(''.join(lines), abs_path)
                
    best_match = None
    smallest_size = float('inf')
    for node in ast.walk(tree):
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if hasattr(node, 'decorator_list') and node.decorator_list:
                start_line = node.decorator_list[0].lineno
            else:
                start_line = node.lineno

            end_line = node.end_lineno
            
            if start_line <= target <= end_line:
                if isinstance(node, NODE_TYPES):
                    size = end_line - start_line

                    if size < smallest_size:
                        smallest_size = size
                        best_match = node
    if best_match is None:
        return None

    best_match.filename = abs_path
    return best_match