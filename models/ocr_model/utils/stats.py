import os
import psutil
import subprocess
import statistics

def get_process_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_gpu_memory_nvidia_smi_mb():
    """Get GPU memory via nvidia-smi (works for all frameworks)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0
    
def compute_stats(latencies): 
    n = len(latencies)
    mean = statistics.mean(latencies)
      
    return {
        "mean_ms":       mean * 1000,
        "median_ms":     statistics.median(latencies) * 1000,
        "stdev_ms":      (statistics.stdev(latencies) * 1000) if n >= 2 else 0.0,
        "min_ms":        min(latencies) * 1000,
        "max_ms":        max(latencies) * 1000,
        "p95_ms":        sorted(latencies)[int(n * 0.95)] * 1000,
        "throughput_fps": 1.0 / mean,
    }