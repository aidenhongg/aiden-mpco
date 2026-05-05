from github import Github, GithubException
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import subprocess
import platform
import tomlkit
import shutil
import json
import stat
import os
import re

from src.profiler.profile import ProjProfile

from dotenv import load_dotenv
load_dotenv()
# TODO: remove redundant make venv step and just call from pyProj instead

REPOS_PATH = Path('./repos.json')
PROJECT_INFO_PATH = Path('./prompts/project_info.json')
PROJECTS_DIR = Path('./repos/projects')
VENVS_DIR = Path('./repos/venvs')

CREATED_AFTER = '2025-12-01'
MIN_STARS = 100

def _remove_readonly(func, path, _):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _clone_repo(author, name):
    repo_path = PROJECTS_DIR / name
    if repo_path.is_dir():
        shutil.rmtree(repo_path, onexc=_remove_readonly)

    try:
        subprocess.run(
            ['git', 'clone', f"https://github.com/{author}/{name}.git"],
            cwd=PROJECTS_DIR, check=True,
        )

        toml_path = repo_path / 'pyproject.toml'
        with open(toml_path, "r", encoding="utf-8") as f:
            config = tomlkit.load(f)

        ini_options = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
        if not ini_options:
            raise ValueError("No pytest ini_options found in pyproject.toml")

        testpaths = ini_options.get("testpaths", [])
        if isinstance(testpaths, list) and len(testpaths) > 0:
            primary_path = testpaths[0]
            if not (repo_path / primary_path).exists():
                raise ValueError(f"Primary testpath '{primary_path}' does not exist.")
            testpaths.clear()
            testpaths.append(primary_path)

        if "addopts" in ini_options:
            del ini_options["addopts"]

        if platform.system() == "Windows":
            timeout_method = ini_options.get("timeout_method")
            if isinstance(timeout_method, str):
                ini_options["timeout_method"] = "thread"

        with open(toml_path, "w", encoding="utf-8") as f:
            tomlkit.dump(config, f)

    except Exception:
        if repo_path.is_dir():
            shutil.rmtree(repo_path, onexc=_remove_readonly)
        raise


def _run_cmd(cmd, cwd=None, shell=False, check=True):
    str_cmd = [str(arg) for arg in cmd] if not shell else cmd
    return subprocess.run(str_cmd, cwd=cwd, check=check, capture_output=False, shell=shell)


def _parse_missing_modules(report):
    tree = ET.parse(report)
    pattern = re.compile(r"ModuleNotFoundError: No module named '([^']+)'|ImportError: No module named ([^ \n]+)")
    modules = []

    for testcase in tree.getroot().iter('testcase'):
        for issue in testcase.findall('error') + testcase.findall('failure'):
            if issue.text and (match := pattern.search(issue.text)):
                modules.append((match.group(1) or match.group(2)).split('.')[0])

    return modules

def _create_venv(name):
    venv_path = VENVS_DIR.resolve() / name

    if venv_path.exists():
        return

    repo_path = PROJECTS_DIR.resolve() / name
    cache_path = VENVS_DIR.resolve() / 'pip_cache'
    report_path = Path('./src/profiler/temp').resolve() / 'setup_report.xml'

    if platform.system() == "Windows":
        venvpy = venv_path / "Scripts" / "python.exe"
        activate = str(venv_path / 'Scripts' / 'activate.bat')
    else:
        venvpy = venv_path / "bin" / "python"
        activate = f'source {venv_path / "bin" / "activate"}'

    _run_cmd(['uv', 'venv', venv_path], cwd=repo_path)
    _run_cmd(" ".join((
        activate,
        f"&& uv sync --active --project {repo_path} --cache-dir {cache_path} --all-groups",
        "&& python -m ensurepip --upgrade && python -m pip install pytest"
    )), shell=True)

    result = _run_cmd([venvpy, '-m', 'pytest', f"--junitxml={report_path}"], cwd=repo_path, check=False)
    if result.returncode > 1:
        modules = _parse_missing_modules(report_path)
        _run_cmd([venvpy, '-m', 'pip', 'install', *modules])
        result = _run_cmd([venvpy, '-m', 'pytest'], cwd=repo_path, check=False)
        if result.returncode > 1:
            raise RuntimeError(f"Tests failed to run in {name}")


def _cleanup(name):
    for path in (PROJECTS_DIR / name, VENVS_DIR / name):
        if path.is_dir():
            shutil.rmtree(path, onexc=_remove_readonly)


def _fetch_project_info(gh, author, name):
    repo = gh.get_repo(f"{author}/{name}")
    return {
        'description': repo.description,
        'languages': ', '.join(repo.get_languages().keys())
    }


def _setup_repo(gh, author, name):
    _clone_repo(author, name)
    _create_venv(name)
    profile = ProjProfile(name)
    profile.initialize()
    if len(profile.top_bottlenecks) <= 6:
        raise ValueError(f"Insufficient bottlenecks ({len(profile.top_bottlenecks)})")
    return _fetch_project_info(gh, author, name)


def main():
    gh = Github(os.environ["GITHUB_KEY"])

    with open(REPOS_PATH, 'r') as f:
        repos = json.load(f)

    project_info = {}
    for name, author in repos.items():
        print(f"[{name}] Setting up {author}/{name}...")
        try:
            project_info[name] = _setup_repo(gh, author, name)
            print(f"[{name}] Done")
        except Exception as e:
            print(f"Skipping {author}/{name}: {e}")
            _cleanup(name)

    with open(PROJECT_INFO_PATH, 'w') as f:
        json.dump(project_info, f, indent=4)

    print(f"Set up {len(project_info)} repos from {REPOS_PATH}")


def sample_main(n=10):
    gh = Github(os.environ["GITHUB_KEY"])

    query = f"language:python stars:>{MIN_STARS} created:>{CREATED_AFTER}"
    results = gh.search_repositories(query=query, sort='stars', order='desc')

    repos = {}
    project_info = {}

    for repo in results:
        if len(repos) >= n:
            break

        full_name = repo.full_name
        author, name = full_name.split('/', 1)

        try:
            repo.get_contents("pyproject.toml")
        except GithubException:
            print(f"Skipping {full_name}: no pyproject.toml")
            continue

        try:
            project_info[name] = _setup_repo(gh, author, name)
            repos[name] = author
            print(f"[{len(repos)}/{n}] Added {full_name}")
        except Exception as e:
            print(f"Skipping {full_name}: {e}")
            _cleanup(name)

    with open(REPOS_PATH, 'w') as f:
        json.dump(repos, f, indent=4)

    with open(PROJECT_INFO_PATH, 'w') as f:
        json.dump(project_info, f, indent=4)

    print(f"Collected {len(repos)} repos -> {REPOS_PATH}, {PROJECT_INFO_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, nargs='?', const=None, default=False)
    args = parser.parse_args()

    if not args.s:
        main()
    else:
        sample_main(args.s)