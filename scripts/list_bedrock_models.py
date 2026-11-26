"""
Discover which Bedrock text models this AWS account/region can actually invoke.

Bedrock availability depends on (a) the region, (b) per-model access grants on the
account, and (c) whether the model needs a cross-region inference profile (the
"us." / "eu." prefixed IDs). A model can appear in the catalogue and still fail at
invoke time, so this script optionally sends a 1-token Converse call to each
candidate and reports what came back.

Usage:
    python scripts/list_bedrock_models.py                     # list catalogue
    python scripts/list_bedrock_models.py --probe             # + smoke-test each
    python scripts/list_bedrock_models.py --probe --filter gpt
    python scripts/list_bedrock_models.py --probe --emit      # print MODELS entries

Needs AWS credentials with bedrock:ListFoundationModels,
bedrock:ListInferenceProfiles and bedrock-runtime:Converse.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.environ.get("AWS_REGION", "us-east-2")

# Suggested short keys for config.MODELS, so output is paste-ready.
SUGGESTED_KEY = {
    "openai.gpt-oss-120b-1:0": "gpt_oss_120b",
    "openai.gpt-oss-20b-1:0":  "gpt_oss_20b",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen235b",
    "us.meta.llama3-3-70b-instruct-v1:0": "llama70b3_3",
    "us.meta.llama3-1-8b-instruct-v1:0": "llama8b3_1",
    "us.deepseek.r1-v1:0": "deepseekr1",
}


def _short_key(model_id: str) -> str:
    if model_id in SUGGESTED_KEY:
        return SUGGESTED_KEY[model_id]
    tail = model_id.split(".", 1)[-1]
    for suffix in ("-v1:0", ":0"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
    return tail.replace("-", "_").replace(".", "_").replace(":", "_")


def catalogue(region: str) -> list[dict]:
    """Text-output foundation models visible to this account, plus inference profiles."""
    bedrock = boto3.client("bedrock", region_name=region)
    out: list[dict] = []

    try:
        resp = bedrock.list_foundation_models(byOutputModality="TEXT")
    except NoCredentialsError:
        sys.exit("No AWS credentials found. Configure them, then re-run.")
    except ClientError as e:
        sys.exit(f"ListFoundationModels failed: {e}")

    for m in resp.get("modelSummaries", []):
        if "ON_DEMAND" not in (m.get("inferenceTypesSupported") or []) and \
           "INFERENCE_PROFILE" not in (m.get("inferenceTypesSupported") or []):
            continue
        out.append({
            "id": m["modelId"],
            "provider": m.get("providerName", "?"),
            "name": m.get("modelName", "?"),
            "kind": "foundation",
        })

    # Cross-region inference profiles: these are what you invoke for "us."-prefixed models.
    try:
        prof = bedrock.list_inference_profiles()
        for p in prof.get("inferenceProfileSummaries", []):
            out.append({
                "id": p["inferenceProfileId"],
                "provider": "(profile)",
                "name": p.get("inferenceProfileName", "?"),
                "kind": "profile",
            })
    except ClientError:
        pass  # permission to list profiles is optional

    seen, uniq = set(), []
    for m in out:
        if m["id"] not in seen:
            seen.add(m["id"])
            uniq.append(m)
    return uniq


def probe(model_id: str, region: str) -> tuple[str, str]:
    """Send a minimal Converse call. Returns (status, detail)."""
    rt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(retries={"max_attempts": 1}, connect_timeout=15, read_timeout=60),
    )
    params = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}],
        "inferenceConfig": {"maxTokens": 8, "temperature": 0.0, "topP": 0.9},
    }
    try:
        rt.converse(**params)
        return "OK", "converse ok"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        msg = e.response.get("Error", {}).get("Message", str(e))
        # A model that rejects temperature/topP is still usable — flag it so the
        # caller knows to strip those fields rather than treating it as unavailable.
        if "temperature" in msg or "topP" in msg or "inferenceConfig" in msg:
            return "OK*", f"usable, adjust inferenceConfig: {msg[:90]}"
        if code in ("AccessDeniedException",):
            return "NO-ACCESS", "model access not granted in this account/region"
        if code in ("ValidationException",) and "inference profile" in msg.lower():
            return "NEEDS-PROFILE", "invoke the us./eu. inference-profile ID instead"
        return code, msg[:110]
    except Exception as e:  # noqa: BLE001
        return "ERROR", str(e)[:110]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=REGION, help=f"AWS region (default {REGION})")
    ap.add_argument("--filter", default="", help="substring match on model id/name")
    ap.add_argument("--probe", action="store_true", help="smoke-test each model with Converse")
    ap.add_argument("--emit", action="store_true", help="print paste-ready config.MODELS entries")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    models = catalogue(args.region)
    if args.filter:
        f = args.filter.lower()
        models = [m for m in models if f in m["id"].lower() or f in m["name"].lower()]
    if not models:
        sys.exit(f"No text models matched in {args.region}.")

    print(f"{len(models)} text model(s) in {args.region}\n")

    if not args.probe:
        print(f"{'MODEL ID':<62}{'PROVIDER':<16}NAME")
        for m in sorted(models, key=lambda x: x["id"]):
            print(f"{m['id']:<62}{m['provider']:<16}{m['name']}")
        print("\nRe-run with --probe to see which of these you can actually invoke.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        statuses = list(ex.map(lambda m: probe(m["id"], args.region), models))

    rows = sorted(zip(models, statuses), key=lambda z: (z[1][0] != "OK", z[0]["id"]))
    print(f"{'MODEL ID':<62}{'STATUS':<15}DETAIL")
    for m, (status, detail) in rows:
        print(f"{m['id']:<62}{status:<15}{detail}")

    usable = [m for m, (s, _) in rows if s.startswith("OK")]
    print(f"\n{len(usable)} invokable.")

    if args.emit and usable:
        print("\n# Paste into MODELS in config.py:")
        for m in usable:
            print(f'    "{_short_key(m["id"])}": "{m["id"]}",')


if __name__ == "__main__":
    main()
