"""Verify AWS identity, Bedrock model availability, and one real model invocation.

Run from the project folder after `aws sso login` (or `aws configure`):

    python scripts/check_bedrock.py --region us-east-1

It never prints access keys. The account ID is masked. Results are also written to
docs/evidence/bedrock_check.json so you can commit real (redacted) evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import boto3  # noqa: E402
from botocore.exceptions import ClientError, NoCredentialsError  # noqa: E402

from bedrock_client import DEFAULT_MODEL_ID, BedrockAssistant, BedrockInvocationError  # noqa: E402

TEST_PROMPT = "In two sentences, explain what Amazon S3 is used for."


def mask(account_id: str) -> str:
    return "********" + account_id[-4:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--skip-invoke", action="store_true",
                        help="Only check identity and model list (no billable call).")
    args = parser.parse_args()
    evidence: dict = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "region": args.region, "model_id": args.model_id}

    # 1. Who am I? (free call; shows which identity boto3 is using)
    try:
        ident = boto3.client("sts", region_name=args.region).get_caller_identity()
    except NoCredentialsError:
        print("FAIL: no AWS credentials found. Run `aws sso login` or `aws configure`.")
        return 1
    except ClientError as exc:
        print(f"FAIL: identity check failed: {exc.response['Error']['Code']}")
        return 1
    principal = ident["Arn"].split(":", 5)[-1]
    is_root = principal == "root"
    evidence["account"] = mask(ident["Account"])
    evidence["principal_type"] = principal.split("/")[0]
    print(f"OK   identity: account {evidence['account']}, principal type {evidence['principal_type']}")
    if is_root:
        print("WARN you are using the ROOT user. Create an IAM Identity Center user instead before continuing.")

    # 2. Which text models exist in this region? (free call)
    bedrock = boto3.client("bedrock", region_name=args.region)
    try:
        models = bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
        nova = sorted(m["modelId"] for m in models if m["modelId"].startswith("amazon.nova"))
        evidence["nova_text_models_in_region"] = nova
        print(f"OK   {len(models)} text models listed in {args.region}; Nova: {', '.join(nova) or 'none'}")
    except ClientError as exc:
        print(f"WARN could not list models: {exc.response['Error']['Code']} (needs bedrock:ListFoundationModels)")
    try:
        profiles = bedrock.list_inference_profiles()["inferenceProfileSummaries"]
        ids = sorted(p["inferenceProfileId"] for p in profiles)
        evidence["selected_model_is_inference_profile"] = args.model_id in ids
        print(f"OK   selected ID is {'an' if args.model_id in ids else 'NOT an'} inference profile in this region")
    except ClientError as exc:
        print(f"WARN could not list inference profiles: {exc.response['Error']['Code']}")

    # 3. One real invocation (billable, typically a fraction of a cent for Nova Micro)
    if not args.skip_invoke:
        try:
            answer = BedrockAssistant.from_environment(model_id=args.model_id,
                                                       region=args.region).ask(TEST_PROMPT)
        except BedrockInvocationError as exc:
            print(f"FAIL invocation: {exc.code}: {exc.user_message}")
            evidence["invocation"] = {"ok": False, "error_code": exc.code}
            _write(evidence)
            return 1
        evidence["invocation"] = {"ok": True, "prompt": TEST_PROMPT, "response": answer.text,
                                  "latency_ms": answer.latency_ms, "input_tokens": answer.input_tokens,
                                  "output_tokens": answer.output_tokens, "stop_reason": answer.stop_reason}
        print(f"OK   invocation: {answer.latency_ms} ms, {answer.input_tokens} in / {answer.output_tokens} out tokens")
        print(f"     response: {answer.text[:300]}")
    _write(evidence)
    return 0


def _write(evidence: dict) -> None:
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "evidence" / "bedrock_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"     evidence saved to {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    raise SystemExit(main())
