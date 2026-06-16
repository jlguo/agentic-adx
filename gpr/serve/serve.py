"""
vLLM serving configuration for GPR model.
Serves the GPR model with structured output heads for CTR/CVR/eCPM inference.
"""

import argparse
import json
import os
import sys


VLLM_CONFIG = {
    "model": "",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "quantization": "fp8",
    "dtype": "float16",
    "enforce_eager": True,
    "disable_log_requests": True,
    "max_num_seqs": 64,
    "api_key": "adx-gpr-serve",
}


def generate_vllm_args(model_path: str, port: int = 8000, quant: str = "fp8") -> list:
    return [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--quantization", quant,
        "--dtype", "float16",
        "--enforce-eager",
        "--disable-log-requests",
        "--max-num-seqs", "64",
    ]


def create_vllm_dockerfile():
    return """FROM vllm/vllm-openai:latest

ENV VLLM_API_KEY=adx-gpr-serve
ENV VLLM_MAX_MODEL_LEN=4096
ENV VLLM_GPU_MEMORY_UTILIZATION=0.85

COPY ./gpr_pretrained.pt /model/gpr_pretrained.pt
COPY ./gpr/model/gpr_model.py /model/gpr_model.py

EXPOSE 8000

CMD ["--model", "/model", "--port", "8000", "--quantization", "fp8", "--dtype", "float16", "--enforce-eager"]
"""


def main():
    parser = argparse.ArgumentParser(description="GPR vLLM serving setup")
    parser.add_argument("--model-path", default="/model/gpr_pretrained.pt",
                        help="Path to trained GPR model")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM server port")
    parser.add_argument("--quant", default="fp8",
                        choices=["fp8", "int8", "int4", "none"],
                        help="Quantization method")
    parser.add_argument("--generate-dockerfile", action="store_true",
                        help="Generate Dockerfile for vLLM serving")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate vLLM config JSON")
    args = parser.parse_args()

    if args.generate_dockerfile:
        dockerfile = create_vllm_dockerfile()
        path = os.path.join(os.path.dirname(__file__), "Dockerfile")
        with open(path, "w") as f:
            f.write(dockerfile)
        print(f"Dockerfile written to {path}")

    if args.generate_config:
        config = dict(VLLM_CONFIG)
        config["model"] = args.model_path
        config["quantization"] = args.quant
        print(json.dumps(config, indent=2))

    cmd_args = generate_vllm_args(args.model_path, args.port, args.quant)
    print("\nvLLM launch command:")
    print(" \\\n  ".join(cmd_args))


if __name__ == "__main__":
    main()
