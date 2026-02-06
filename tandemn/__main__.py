"""CLI entry point: python -m tandemn deploy|destroy|status ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from tandemn.models import DeployRequest


def cmd_deploy(args: argparse.Namespace) -> None:
    # Lazy import so --help is fast and doesn't pull in heavy deps
    from tandemn.orchestrator import launch_hybrid

    # Import modal provider to trigger registration
    import tandemn.providers.modal_provider  # noqa: F401

    request = DeployRequest(
        model_name=args.model,
        gpu=args.gpu,
        gpu_count=args.gpu_count,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        serverless_provider=args.serverless_provider,
        spots_cloud=args.spots_cloud,
        region=args.region,
        concurrency=args.concurrency,
        cold_start_mode=args.cold_start_mode,
        scale_to_zero=not args.no_scale_to_zero,
        service_name=args.service_name,
    )

    print(f"Deploying {request.model_name} on {request.gpu}")
    print(f"Service name: {request.service_name}")
    print(f"Serverless provider: {request.serverless_provider}")
    print(f"Spot cloud: {request.spots_cloud}")
    print()

    result = launch_hybrid(request)

    print()
    print("=" * 60)
    print("DEPLOYMENT RESULT")
    print("=" * 60)

    if result.router and result.router.endpoint_url:
        print(f"  Router:     {result.router.endpoint_url}")
    elif result.router and result.router.error:
        print(f"  Router:     FAILED - {result.router.error}")

    if result.serverless and result.serverless.endpoint_url:
        print(f"  Serverless: {result.serverless.endpoint_url}")
    elif result.serverless and result.serverless.error:
        print(f"  Serverless: FAILED - {result.serverless.error}")

    if result.spot and result.spot.endpoint_url:
        print(f"  Spot:       {result.spot.endpoint_url}")
    else:
        print(f"  Spot:       launching in background...")

    print()
    if result.router_url:
        print(f"All traffic -> {result.router_url}")
    print("=" * 60)


def cmd_destroy(args: argparse.Namespace) -> None:
    from tandemn.orchestrator import destroy_hybrid

    print(f"Destroying deployment: {args.service_name}")
    destroy_hybrid(args.service_name)
    print("Done.")


def cmd_status(args: argparse.Namespace) -> None:
    from tandemn.orchestrator import status_hybrid

    status = status_hybrid(args.service_name)
    print(json.dumps(status, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tandemn",
        description="Hybrid GPU Inference Orchestrator",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- deploy --
    p_deploy = subparsers.add_parser("deploy", help="Deploy a model")
    p_deploy.add_argument("--model", required=True, help="Model name (e.g. Qwen/Qwen3-0.6B)")
    p_deploy.add_argument("--gpu", required=True, help="GPU type (e.g. L40S, A100, H100)")
    p_deploy.add_argument("--gpu-count", type=int, default=1)
    p_deploy.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    p_deploy.add_argument("--max-model-len", type=int, default=4096)
    p_deploy.add_argument("--serverless-provider", default="modal", help="Serverless backend")
    p_deploy.add_argument("--spots-cloud", default="aws", help="Cloud for spot GPUs")
    p_deploy.add_argument("--region", default=None)
    p_deploy.add_argument("--concurrency", type=int, default=32)
    p_deploy.add_argument("--cold-start-mode", default="fast_boot", choices=["fast_boot", "no_fast_boot"])
    p_deploy.add_argument("--no-scale-to-zero", action="store_true")
    p_deploy.add_argument("--service-name", default=None, help="Custom service name")
    p_deploy.set_defaults(func=cmd_deploy)

    # -- destroy --
    p_destroy = subparsers.add_parser("destroy", help="Tear down a deployment")
    p_destroy.add_argument("--service-name", required=True)
    p_destroy.set_defaults(func=cmd_destroy)

    # -- status --
    p_status = subparsers.add_parser("status", help="Check deployment status")
    p_status.add_argument("--service-name", required=True)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
