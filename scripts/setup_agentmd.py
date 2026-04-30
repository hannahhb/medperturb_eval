"""
One-time setup script for AgentMD evaluation.

Downloads riskcalcs.json from the ncbi-nlp/Clinical-Tool-Learning repository
and saves it to data/agentmd/riskcalcs.json.

Usage:
    python scripts/setup_agentmd.py
    python scripts/setup_agentmd.py --data-dir /custom/path
"""
import argparse
import json
import os
import sys
import urllib.request

# The riskcalcs.json is published via Zenodo alongside the GitHub repo.
# We download directly from the GitHub LFS / raw URL.
RISKCALCS_URLS = [
    # Try GitHub raw first
    "https://raw.githubusercontent.com/ncbi-nlp/Clinical-Tool-Learning/main/riskqa_evaluation/tools/riskcalcs.json",
    # Zenodo DOI redirect fallback (user may need to download manually)
]

MANUAL_INSTRUCTIONS = """
Automatic download failed. Please download riskcalcs.json manually:

  1. Go to: https://github.com/ncbi-nlp/Clinical-Tool-Learning
  2. Navigate to: riskqa_evaluation/tools/riskcalcs.json
  3. Click "Raw" and save the file to:
       {dest_path}

Or clone the repo and copy the file:
  git clone https://github.com/ncbi-nlp/Clinical-Tool-Learning.git
  cp Clinical-Tool-Learning/riskqa_evaluation/tools/riskcalcs.json {dest_path}
"""


def download_riskcalcs(dest_path: str) -> bool:
    for url in RISKCALCS_URLS:
        print(f"  Trying: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            # Validate it is valid JSON
            parsed = json.loads(data)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            n = len(parsed) if isinstance(parsed, (list, dict)) else "?"
            print(f"  Saved {n} tools → {dest_path}")
            return True
        except Exception as exc:
            print(f"  Failed ({exc})")
    return False


def check_dependencies():
    missing = []
    for pkg, import_name in [("faiss-cpu", "faiss"), ("transformers", "transformers"), ("torch", "torch")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Set up AgentMD for MedPerturb evaluation")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "agentmd"),
        help="Directory to save AgentMD data files (default: data/agentmd/)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    dest_path = os.path.join(data_dir, "riskcalcs.json")

    print("=== AgentMD Setup ===\n")

    # 1. Check dependencies
    print("1. Checking dependencies …")
    missing = check_dependencies()
    if missing:
        print(f"   Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        if "faiss" in " ".join(missing):
            print("   Note: use faiss-gpu instead of faiss-cpu if you have a GPU")
    else:
        print("   All dependencies present.")

    # 2. Download riskcalcs.json
    print(f"\n2. Downloading riskcalcs.json → {dest_path}")
    if os.path.exists(dest_path):
        print(f"   Already exists, skipping. Delete {dest_path} to re-download.")
    else:
        ok = download_riskcalcs(dest_path)
        if not ok:
            print(MANUAL_INSTRUCTIONS.format(dest_path=dest_path))
            sys.exit(1)

    # 3. Validate
    print("\n3. Validating riskcalcs.json …")
    with open(dest_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        n = len(data)
        has_code = sum(1 for t in data if t.get("code") or t.get("script") or t.get("body"))
    elif isinstance(data, dict):
        n = len(data)
        has_code = sum(1 for t in data.values() if t.get("code") or t.get("script") or t.get("body"))
    else:
        print("   WARNING: unexpected format in riskcalcs.json")
        n, has_code = 0, 0
    print(f"   {n} tools loaded, {has_code} have executable code.")

    # 4. LLM credentials check
    print("\n4. LLM credentials check …")
    if os.environ.get("OPENAI_API_KEY"):
        print("   OPENAI_API_KEY found — AgentMD will use OpenAI as LLM backend.")
    elif os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        print("   AWS credentials found — AgentMD will use Bedrock as LLM backend.")
    else:
        print("   WARNING: No LLM credentials found.")
        print("   Set OPENAI_API_KEY (for OpenAI) or configure AWS credentials (for Bedrock).")

    print("\n=== Setup complete ===")
    print("\nRun AgentMD evaluation:")
    print("  python run.py --mode AGENTMD --name llama70b3_3 --datasets oncqa askdocs")
    print("  python run.py --mode AGENTMD --name llama70b3_3 --debug  # quick test")


if __name__ == "__main__":
    main()
