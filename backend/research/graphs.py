import os
import json
from typing import List, Dict

try:
    import matplotlib
    # Use non-gui backend for server environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except Exception as e:
    PLOT_AVAILABLE = False
    print(f"Graph Generation Disabled: {e}")

class GraphGenerator:
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir

    def generate_all(self, data: List[Dict]):
        if not PLOT_AVAILABLE:
            print("Skipping graph generation (matplotlib missing).")
            return
        self.plot_execution_time(data)
        self.plot_success_rate(data)

    def plot_execution_time(self, data):
        try:
            agent_times = [d['metrics']['execution_time'] for d in data if d['mode'] == 'agent']
            baseline_times = [d['metrics']['execution_time'] for d in data if d['mode'] == 'baseline']
            
            plt.figure(figsize=(10, 6))
            plt.boxplot([baseline_times, agent_times], labels=['Baseline', 'Agent'])
            plt.title('Execution Time Distribution')
            plt.ylabel('Time (s)')
            plt.savefig(os.path.join(self.output_dir, 'execution_time_boxplot.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Graph Error: {e}")

    def plot_success_rate(self, data):
        try:
            modes = ['Base', 'Agent']
            agent_suc = len([d for d in data if d['mode'] == 'agent' and d['metrics'].get('success')])
            base_suc = len([d for d in data if d['mode'] == 'baseline' and d['metrics'].get('success')])
            
            total = len([d for d in data if d['mode'] == 'agent'])
            if total == 0: return

            rates = [base_suc/total * 100, agent_suc/total * 100]
            
            plt.figure(figsize=(8, 6))
            plt.bar(modes, rates, color=['gray', 'blue'])
            plt.title('Success Rate Comparison')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.savefig(os.path.join(self.output_dir, 'success_rate.png'), dpi=300)
            plt.close()
        except Exception as e:
             print(f"Graph Error: {e}")

    def generate_single_run(self, metrics: Dict, run_dir: str):
        if not PLOT_AVAILABLE:
            return

        log_dir = os.path.join(run_dir, "evaluation", "graphs")
        os.makedirs(log_dir, exist_ok=True)

        # 1. Phase Duration
        try:
            phases = metrics.get("phases", {})
            if phases:
                names = list(phases.keys())
                times = [p.get("duration", 0) for p in phases.values()]
                
                plt.figure(figsize=(10, 5))
                y_pos = np.arange(len(names))
                plt.barh(y_pos, times, color='skyblue')
                plt.yticks(y_pos, names)
                plt.xlabel('Seconds')
                plt.title('Phase Duration')
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, "phase_duration.png"))
                plt.close()
        except Exception as e:
            print(f"Graph Error (Phases): {e}")

        # 2. Token Cost
        try:
            usage = metrics.get("token_usage", {})
            # filter out total_estimated_cost and keep keys
            labels = [k for k in usage.keys() if k != "total_estimated_cost" and usage[k] > 0]
            sizes = [usage[k] for k in labels]
            
            if sum(sizes) > 0:
                plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
                plt.title('Token Usage Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, "token_usage.png"))
                plt.close()
        except Exception as e:
             print(f"Graph Error (Tokens): {e}")

if __name__ == "__main__":
    if os.path.exists("evaluation_results/benchmark_summary.json"):
        with open("evaluation_results/benchmark_summary.json") as f:
            data = json.load(f)
            g = GraphGenerator()
            g.generate_all(data)
