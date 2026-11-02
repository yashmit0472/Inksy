"""Read-only compute inventory and tiny CUDA arithmetic probe; never trains a model."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from ml.utils import write_json


def inspect_runtime():
    report = {
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "training_performed": False,
        "windows_settings_changed": False,
        "packages_changed": False,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        report["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["nvidia_smi_error"] = str(error)
    if report["cuda_available"]:
        try:
            with torch.inference_mode():
                tensor = torch.ones(16, 16, device="cuda")
                result = tensor @ tensor
                torch.cuda.synchronize()
                report["cuda_arithmetic_passed"] = bool((result == 16).all())
                report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            del tensor, result
        except (RuntimeError, OSError) as error:
            report["cuda_arithmetic_passed"] = False
            report["cuda_probe_error"] = str(error)
    else:
        report["cuda_arithmetic_passed"] = False
    report["gpu_training_environment_ready"] = report["cuda_arithmetic_passed"]
    report["training_memory_budget_verified"] = False
    report["next_step"] = (
        "Measure the proposed model's memory/time in an owner-run bounded smoke test."
        if report["cuda_arithmetic_passed"]
        else "GPU hardware alone is insufficient. Prepare a separate compatible CUDA environment "
        "and retain the working CPU environment; training feasibility is still unmeasured."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime-check.json"))
    args = parser.parse_args()
    report = inspect_runtime()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
