import os
import sys
import json
import time
import shutil
import statistics
import threading
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Add backend directory to path to support 'from services import ...' legacy style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from backend.simple_builder import SDLCBuilder
# Import wrapper strictly for ablation simulation if needed, but wrapper imports evaluate, so be careful.
# We import wrapper locally where needed or assume it's available.
# from backend.research.wrapper import ResearchWrapper # Avoid top-level if circular
import backend.services.adapters as adapters
from backend.research.graphs import GraphGenerator

# ------------------------------------------------------------------------------
# BENCHMARK TASKS
# ------------------------------------------------------------------------------
TASKS = [
    {
        "id": "task_001",
        "name": "Simple Calculator",
        "prompt": "Create a simple calculator web app with add, subtract, multiply, divide."
    },
    {
        "id": "task_002", 
        "name": "Todo List",
        "prompt": "Create a Todo List app where I can add, delete, and mark items as done."
    }
]

# ------------------------------------------------------------------------------
# BASELINE RUNNER
# ------------------------------------------------------------------------------
def run_baseline(prompt: str, run_dir: str) -> Dict[str, Any]:
    """
    Executes a single-shot LLM call to simulate a non-agentic baseline.
    Silent execution.
    """
    start = time.time()
    # Writes to evaluation/ inside run_dir to keep run clean
    output_path = os.path.join(run_dir, "evaluation", "baseline_output.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check cache/existing
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            content = f.read()
        return {
            "success": True,
            "execution_time": 0.5, # Cached
            "cached": True,
            "content_length": len(content)
        }

    try:
        # Using Mistral as the baseline model
        response = adapters.call_mistral(f"Generate code for: {prompt}. Output only code.", "mistral-small-latest")
        with open(output_path, "w") as f:
            f.write(response)
        success = True
    except Exception as e:
        print(f"[Baseline] Failed: {e}")
        success = False
        
    duration = time.time() - start
    return {
        "success": success,
        "execution_time": duration,
        "iterations": 1,
        "phases": {"baseline": {"status": "completed" if success else "failed"}}
    }

# ------------------------------------------------------------------------------
# ABLATION SIMULATOR
# ------------------------------------------------------------------------------
def run_ablation_simulation(run_dir: str, prompt: str, original_metrics: Dict) -> Dict[str, Any]:
    """
    Simulates ablation configurations.
    """
    memory_impact = {
        "config": "memory_off",
        "estimated_success_prob": 0.7 if original_metrics.get("success") else 0.0,
        "notes": "Simulated removal of context retention."
    }
    sandbox_impact = {
        "config": "sandbox_off",
        "estimated_risk": "High",
        "notes": "Code execution would be unchecked."
    }
    return {
        "memory_off": memory_impact,
        "sandbox_off": sandbox_impact
    }

# ------------------------------------------------------------------------------
# AUTO-EVALUATION PIPELINE (SINGLE RUN)
# ------------------------------------------------------------------------------
def evaluate_run(run_dir: str, prompt: str):
    """
    Main entry point triggered after a live run completes.
    """
    print(f"[AutoEval] Starting evaluation for {run_dir}")
    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Load Agent Metrics
    metrics_path = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[AutoEval] No metrics.json found. Skipping.")
        return

    try:
        with open(metrics_path, "r") as f:
            agent_metrics = json.load(f)
    except:
        return

    # 2. Run Baseline
    baseline_metrics = run_baseline(prompt, run_dir)
    with open(os.path.join(eval_dir, "baseline.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)
        
    # Also save as CSV as requested
    try:
        import csv
        with open(os.path.join(eval_dir, "baseline.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric", "value"])
            for k, v in baseline_metrics.items():
                if isinstance(v, (int, float, bool, str)):
                    writer.writerow([k, v])
    except: pass

    # 3. Run Ablation
    ablation_results = run_ablation_simulation(run_dir, prompt, agent_metrics)
    with open(os.path.join(eval_dir, "ablation.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    # 4. Compute Statistics
    stats = {
        "run_id": agent_metrics.get("task_id"),
        "timestamp": datetime.now().isoformat(),
        "agent": {
            "time": agent_metrics.get("execution_time", 0),
            "success": agent_metrics.get("success", False),
            "cost": agent_metrics.get("token_usage", {}).get("total_estimated_cost", 0)
        },
        "baseline": {
            "time": baseline_metrics.get("execution_time", 0),
            "success": baseline_metrics.get("success", False)
        },
        "improvement": {
            "speedup": 0.0
        }
    }
    
    if stats["agent"]["time"] > 0 and stats["baseline"]["time"] > 0:
        stats["improvement"]["speedup"] = stats["baseline"]["time"] / stats["agent"]["time"]

    with open(os.path.join(eval_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # 5. Generate Graphs
    try:
        gg = GraphGenerator(output_dir=eval_dir) # Initialize with output dir for general usage
        gg.generate_single_run(agent_metrics, run_dir)
    except Exception as e:
        print(f"[AutoEval] Graph generation failed: {e}")

    # 6. Final Summary
    summary = {
        "meta": {"generated_at": datetime.now().isoformat()},
        "metrics": agent_metrics,
        "baseline": baseline_metrics,
        "ablation": ablation_results,
        "stats": stats
    }
    with open(os.path.join(eval_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[AutoEval] Completed. Artifacts in {eval_dir}")

# ------------------------------------------------------------------------------
# BENCHMARK EVALUATOR (SUITE)
# ------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def run_agent(self, task: Dict, ablation_config: Dict = None):
        """Runs the actual agent."""
        job_id = f"eval_{task['id']}_{int(time.time())}"
        if ablation_config:
            job_id += "_ablated"
            
        run_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Local import to avoid circular dep
        from backend.research.wrapper import ResearchWrapper
        
        # Setup Builder
        builder = SDLCBuilder(runs_dir=self.output_dir)
        wrapper = ResearchWrapper(builder, enable_metrics=True, enable_protection=True, evaluation_mode=True)
        
        print(f"Running Task: {task['name']} [{'Ablated' if ablation_config else 'Standard'}]")
        
        try:
            wrapper.init_run(task["prompt"], job_id)
            real_run_dir = os.path.join(self.output_dir, job_id) 
            
            wrapper.run_build(real_run_dir, task["prompt"])
            
            # Load metrics from strict evaluation/ location now
            metrics_path = os.path.join(real_run_dir, "evaluation", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    return json.load(f)
            return {"success": False, "error": "Metrics missing"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_benchmark(self):
        print("Starting Benchmark Execution...")
        summary_data = []

        for i, task in enumerate(TASKS):
            print(f"\n--- Task {i+1}/{len(TASKS)}: {task['name']} ---")
            
            # 1. Run Baseline
            base_dir = os.path.join(self.output_dir, f"{task['id']}_baseline")
            os.makedirs(base_dir, exist_ok=True)
            baseline_metrics = run_baseline(task["prompt"], base_dir)
            summary_data.append({
                "task": task["name"],
                "mode": "baseline",
                "metrics": baseline_metrics
            })
            self.save_results(summary_data)
            
            # 2. Run Agent
            agent_metrics = self.run_agent(task)
            summary_data.append({
                "task": task["name"],
                "mode": "agent",
                "metrics": agent_metrics
            })
            self.save_results(summary_data)

        self.generate_stats(summary_data)

    def save_results(self, data):
        with open(os.path.join(self.output_dir, "benchmark_summary.json"), "w") as f:
            json.dump(data, f, indent=2)

    def generate_stats(self, data):
        # Calculate improvement
        agent_runs = [d for d in data if d["mode"] == "agent"]
        baseline_runs = [d for d in data if d["mode"] == "baseline"]
        
        if not agent_runs: return
        
        avg_time_agent = statistics.mean([r["metrics"].get("execution_time", 0) for r in agent_runs])
        avg_time_base = statistics.mean([r["metrics"].get("execution_time", 0) for r in baseline_runs]) if baseline_runs else 0
        
        success_rate_agent = len([r for r in agent_runs if r["metrics"].get("success")]) / len(agent_runs)
        
        stats = {
            "agent_avg_time": avg_time_agent,
            "baseline_avg_time": avg_time_base,
            "agent_success_rate": success_rate_agent,
            "improvement_factor": avg_time_base / avg_time_agent if avg_time_agent > 0 else 0
        }
        
        with open(os.path.join(self.output_dir, "statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
        print("Statistics Generated:", json.dumps(stats, indent=2))
        
        # Generate Graphs
        try:
             gg = GraphGenerator(self.output_dir)
             gg.generate_all(data)
             print("Graphs Generated successfully.")
        except ImportError:
             print("Graph Generation Skipped: Matplotlib not available.")
        except Exception as e:
             print(f"Graph Generation Warning: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        # CLI Mode for Auto-Eval
        evaluate_run(sys.argv[1], sys.argv[2])
    else:
        # Suite Mode
        e = Evaluator()
        e.run_benchmark()
