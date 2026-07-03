# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for Azure OpenAI access to gpt-5.5.

On Azure, `model=` is the *deployment name* you created in the portal, NOT the
base model name. If your deployment of gpt-5.5 is literally named "gpt-5.5",
the default below works; otherwise pass --deployment <your-deployment-name>.

Requires env vars (same ones LLMManager uses):
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
    export AZURE_OPENAI_API_KEY="<key>"

Run:
    python scripts/test_azure_gpt55.py
    python scripts/test_azure_gpt55.py --deployment my-gpt55 --api responses
"""

import argparse
import os
import sys

import openai


def main(args: argparse.Namespace) -> None:
    # --- env check ---
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    missing = [n for n, v in [("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", key)] if not v]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}")

    print(f"endpoint     = {endpoint}")
    print(f"deployment   = {args.deployment}")
    print(f"api_version  = {args.api_version}")
    print(f"api          = {args.api}\n")

    client = openai.AzureOpenAI(api_version=args.api_version)

    try:
        if args.api == "responses":
            # gpt-5 family prefers the responses API (see scripts/quota_check.py)
            resp = client.responses.create(model=args.deployment, input=args.prompt)
            out = resp.output_text
        else:
            resp = client.chat.completions.create(
                model=args.deployment,
                messages=[{"role": "user", "content": args.prompt}],
            )
            out = resp.choices[0].message.content
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e))
        sys.exit(1)

    print("OK. Model replied:\n")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test Azure OpenAI gpt-5.5 access.")
    parser.add_argument("--deployment", default="gpt-5.5", help="Azure deployment name (NOT base model name).")
    parser.add_argument("--api", choices=["responses", "chat"], default="responses", help="Which API to call.")
    parser.add_argument("--api-version", default="2025-04-01-preview", help="Azure OpenAI API version.")
    parser.add_argument("--prompt", default="Say hi in one short sentence.", help="Test prompt.")
    args = parser.parse_args()
    main(args)
