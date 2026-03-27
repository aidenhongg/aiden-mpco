import json
import traceback
from pathlib import Path
from uuid import uuid4

from . import chains, telemetry
from .chains import MetaChain, evaluation
from .profiler.profile import ProjProfile
from .patches.patch import Patch, PatchStack

REPOS_PATH = Path("./repos.json")
RESULTS_PATH = Path("./src/results.json")
MAX_RETRIES = 10


def _save(results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def run():
    with open(REPOS_PATH) as f:
        projects = list(json.load(f).keys())

    prompt_names = list(chains.PROMPTS) + ["meta"]
    results = {}

    # Profile each project once upfront to avoid redundant baseline runs.
    baselines: dict[str, ProjProfile] = {}
    for proj_name in projects:
        profile = ProjProfile(proj_name=proj_name)
        try:
            profile.initialize()
        except RuntimeError as e:
            print(f"Skipping {proj_name} (baseline failed): {e}")
            continue
        baselines[proj_name] = profile

    for agent_name in chains.AGENTS:
        for prompt_name in prompt_names:
            combo_key = f"{agent_name}/{prompt_name}"
            print(f"\n{'#'*60}\nAgent: {combo_key}\n{'#'*60}")
            agent_results = {}
            results[combo_key] = agent_results

            for proj_name in projects:
                if proj_name not in baselines:
                    continue

                print(f"\n{'='*60}\nProject: {proj_name}\n{'='*60}")

                chain = chains.build_chain(agent_name, prompt_name, proj_name)

                profile = baselines[proj_name]

                patch_stack = PatchStack()
                proj_data = {
                    "start_runtime_avg": profile.start_runtime,
                    "end_runtime_avg": None,
                    "snippets": [],
                }
                agent_results[proj_name] = proj_data
                last_runtime = profile.start_runtime

                for snippet in profile.yield_snippet():
                    original_code = snippet.code
                    record = {
                        "original_code": original_code,
                        "optimized_code": original_code,
                        "project_name": proj_name,
                        "failed_regenerations": 0,
                        "runtime_diff": 0.0,
                    }

                    failures = 0
                    success = False
                    patch = None

                    while failures < MAX_RETRIES:
                        optimized = None
                        try:
                            run_id = uuid4()
                            optimized = chains.invoke(
                                chain, original_code, snippet.scope,
                                regenerate=failures > 0, run_id=run_id,
                            )

                            patch = Patch(
                                code_object=snippet._asdict(),
                                optimized_code=optimized,
                                root=str(profile.repo_path),
                            )

                            if not patch.apply_patch():
                                failures += 1
                                continue

                            new_runtime = profile.check_patch()
                            record.update(
                                optimized_code=optimized,
                                runtime_diff=new_runtime - last_runtime,
                                failed_regenerations=failures,
                            )
                            if isinstance(chain, MetaChain):
                                record["generated_prompt"] = chain._cached_prompt
                            record.update(evaluation.score(original_code, optimized))
                            record.update(telemetry.fetch_run_stats(run_id))
                            last_runtime = new_runtime
                            patch_stack.push(patch)
                            success = True
                            break
                        except Exception as e:
                            traceback.print_exc()
                            print(snippet.code)
                            if optimized:
                                print("Failed optimization: ")
                                print(optimized)
                                
                            print(f"{failures+1} failed attempts!")
                            if patch is not None:
                                patch.revert_patch()
                            failures += 1

                    if not success:
                        record["failed_regenerations"] = failures
                        print(f"  Snippet failed after {MAX_RETRIES} retries, skipping")

                    proj_data["snippets"].append(record)
                    _save(results)

                try:
                    proj_data["end_runtime_avg"] = profile.new_average_runtime()
                except RuntimeError:
                    pass

                _save(results)
                patch_stack.revert_all()

    print("\nComplete. Results saved to src/results.json")
    return results
