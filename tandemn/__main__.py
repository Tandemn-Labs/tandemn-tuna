"""CLI entry point: python -m tandemn deploy|destroy|status|list ..."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from tandemn.models import DeployRequest
from tandemn.scaling import default_scaling_policy, load_scaling_policy


def cmd_deploy(args: argparse.Namespace) -> None:
    # Lazy import so --help is fast and doesn't pull in heavy deps
    from tandemn.orchestrator import launch_hybrid
    from tandemn.state import save_deployment

    # Import providers to trigger registration
    import tandemn.providers.cloudrun_provider  # noqa: F401
    import tandemn.providers.modal_provider  # noqa: F401
    import tandemn.providers.runpod_provider  # noqa: F401
    import tandemn.spot.sky_launcher  # noqa: F401

    _setup_gcp_env(args)

    # Build scaling policy: defaults <- YAML <- CLI flags
    if args.scaling_policy:
        scaling = load_scaling_policy(args.scaling_policy)
    else:
        scaling = default_scaling_policy()

    if args.concurrency is not None:
        scaling.serverless.concurrency = args.concurrency
    if args.workers_max is not None:
        scaling.serverless.workers_max = args.workers_max
    if args.no_scale_to_zero:
        scaling.spot.min_replicas = max(1, scaling.spot.min_replicas)
        scaling.serverless.scaledown_window = 300
        scaling.serverless.workers_min = max(1, scaling.serverless.workers_min)

    # Auto-select cheapest serverless provider if not specified
    serverless_provider = args.serverless_provider
    auto_selected = False
    if serverless_provider is None:
        from tandemn.catalog import normalize_gpu_name, query as catalog_query
        try:
            gpu_name = normalize_gpu_name(args.gpu)
        except KeyError:
            gpu_name = args.gpu
        result = catalog_query(gpu=gpu_name)
        cheapest = result.cheapest()
        if cheapest:
            serverless_provider = cheapest.provider
            auto_selected = True
            _print_provider_selection(gpu_name, result, serverless_provider)
        else:
            serverless_provider = "modal"

    request = DeployRequest(
        model_name=args.model,
        gpu=args.gpu,
        gpu_count=args.gpu_count,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        serverless_provider=serverless_provider,
        spots_cloud=args.spots_cloud,
        region=args.region,
        cold_start_mode=args.cold_start_mode,
        scaling=scaling,
        service_name=args.service_name,
        public=args.public,
    )

    print(f"Deploying {request.model_name} on {request.gpu}")
    print(f"Service name: {request.service_name}")
    if not auto_selected:
        print(f"Serverless provider: {request.serverless_provider}")
    print(f"Spot cloud: {request.spots_cloud}")
    print()

    result = launch_hybrid(request)

    # Persist deployment metadata
    save_deployment(request, result)

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
    from tandemn.providers.registry import ensure_providers_for_deployment
    from tandemn.state import load_deployment, update_deployment_status

    record = load_deployment(args.service_name)
    if record is None:
        print(f"Error: no deployment record found for '{args.service_name}'.", file=sys.stderr)
        sys.exit(1)

    ensure_providers_for_deployment(record)

    print(f"Destroying deployment: {args.service_name}")
    destroy_hybrid(args.service_name, record=record)
    update_deployment_status(args.service_name, "destroyed")
    print("Done.")


def cmd_status(args: argparse.Namespace) -> None:
    from tandemn.orchestrator import status_hybrid
    from tandemn.providers.registry import ensure_providers_for_deployment
    from tandemn.state import load_deployment

    record = load_deployment(args.service_name)
    if record is None:
        print(f"Error: no deployment record found for '{args.service_name}'.", file=sys.stderr)
        sys.exit(1)

    ensure_providers_for_deployment(record)

    status = status_hybrid(args.service_name, record=record)
    print(json.dumps(status, indent=2, default=str))


def _setup_gcp_env(args: argparse.Namespace) -> None:
    """Forward GCP-related CLI arguments to environment variables."""
    if getattr(args, "gcp_project", None):
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project
    if getattr(args, "gcp_region", None):
        os.environ["GOOGLE_CLOUD_REGION"] = args.gcp_region


def cmd_check(args: argparse.Namespace) -> None:
    from tandemn.providers.registry import get_provider

    # Import providers to trigger registration
    import tandemn.providers.cloudrun_provider  # noqa: F401
    import tandemn.providers.modal_provider  # noqa: F401
    import tandemn.providers.runpod_provider  # noqa: F401
    import tandemn.spot.sky_launcher  # noqa: F401

    _setup_gcp_env(args)

    provider_name = args.provider
    try:
        provider = get_provider(provider_name)
    except KeyError:
        print(f"Error: unknown provider '{provider_name}'.", file=sys.stderr)
        sys.exit(1)

    # Build a minimal DeployRequest for preflight
    request = DeployRequest(
        model_name="check",
        gpu=args.gpu or "L4",
        region=args.gcp_region,
        serverless_provider=provider_name,
    )

    print(f"Checking {provider_name}...")
    print()

    result = provider.preflight(request)

    for check in result.checks:
        tag = "PASS" if check.passed else "FAIL"
        suffix = ""
        if check.auto_fixed:
            suffix = " (auto-fixed)"
        print(f"  [{tag}] {check.name}: {check.message}{suffix}")
        if not check.passed and check.fix_command:
            print(f"         Fix: {check.fix_command}")

    print()
    if result.ok:
        print(f"{provider_name}: all checks passed.")
    else:
        count = len(result.failed)
        print(f"{provider_name}: {count} check(s) failed.")
        sys.exit(1)


def cmd_show_gpus(args: argparse.Namespace) -> None:
    from tandemn.catalog import (
        fetch_spot_prices,
        get_gpu_spec,
        normalize_gpu_name,
        query,
        GPU_SPECS,
    )

    gpu_filter = None
    if args.gpu:
        try:
            gpu_filter = normalize_gpu_name(args.gpu)
        except KeyError:
            print(f"Error: unknown GPU '{args.gpu}'.", file=sys.stderr)
            sys.exit(1)

    spot_prices: dict = {}
    if args.spot:
        spot_prices = fetch_spot_prices(cloud="aws")

    result = query(gpu=gpu_filter, provider=args.provider)
    result.spot_prices = spot_prices

    if gpu_filter:
        _print_gpu_detail(gpu_filter, result, spot_prices, get_gpu_spec)
    else:
        _print_gpu_table(result, spot_prices, show_spot=args.spot, get_gpu_spec=get_gpu_spec)


def _print_provider_selection(gpu: str, result, selected_provider: str) -> None:
    """Print a compact pricing table showing which provider was auto-selected."""
    from rich.console import Console
    from rich.table import Table
    from tandemn.catalog import get_gpu_spec

    spec = get_gpu_spec(gpu)
    console = Console()

    table = Table(
        title=f"Serverless pricing for {gpu} ({spec.vram_gb} GB)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("")  # checkmark column
    table.add_column("Provider")
    table.add_column("Price", justify="right")

    for entry in result.sorted_by_price():
        if entry.price_per_gpu_hour <= 0:
            continue
        is_selected = entry.provider == selected_provider
        mark = "[bold green]\u2713[/bold green]" if is_selected else " "
        provider_text = (
            f"[bold green]{entry.provider}[/bold green]"
            if is_selected
            else entry.provider
        )
        price_text = (
            f"[bold green]${entry.price_per_gpu_hour:.2f}/hr[/bold green]"
            if is_selected
            else f"${entry.price_per_gpu_hour:.2f}/hr"
        )
        table.add_row(mark, provider_text, price_text)

    console.print(table)
    console.print()


def _format_price(price: float) -> str:
    """Format a price as $X.XX/hr or dash for zero/missing."""
    if price > 0:
        return f"${price:.2f}/hr"
    return "-"


def _print_gpu_detail(gpu: str, result, spot_prices: dict, get_gpu_spec) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    spec = get_gpu_spec(gpu)
    console = Console()

    console.print(f"[bold]GPU:[/bold]  {spec.short_name}")
    console.print(f"[bold]Name:[/bold] {spec.full_name}")
    console.print(f"[bold]VRAM:[/bold] {spec.vram_gb} GB")
    console.print(f"[bold]Arch:[/bold] {spec.arch}")
    console.print()

    table = Table(title="Provider pricing", show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Price", justify="right")
    table.add_column("Details", style="dim")

    all_prices: list[tuple[str, float]] = []

    for entry in result.sorted_by_price():
        extra = ""
        if entry.regions:
            region_list = ", ".join(entry.regions[:3])
            if len(entry.regions) > 3:
                region_list += ", ..."
            extra = region_list
        if entry.price_per_gpu_hour > 0:
            all_prices.append((entry.provider, entry.price_per_gpu_hour))
        table.add_row(entry.provider, _format_price(entry.price_per_gpu_hour), extra)

    if gpu in spot_prices:
        sp = spot_prices[gpu]
        extra = f"{sp.instance_type}, {sp.region}" if sp.instance_type else sp.region
        all_prices.append(("aws spot", sp.price_per_gpu_hour))
        table.add_row("aws spot", _format_price(sp.price_per_gpu_hour), extra)

    console.print(table)

    if all_prices:
        cheapest = min(all_prices, key=lambda x: x[1])
        console.print(
            f"\nCheapest: [bold green]{cheapest[0]}[/bold green] "
            f"at [bold green]${cheapest[1]:.2f}/hr[/bold green]"
        )


def _print_gpu_table(result, spot_prices: dict, show_spot: bool, get_gpu_spec) -> None:
    from rich.console import Console
    from rich.table import Table
    from tandemn.catalog import GPU_SPECS

    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("GPU")
    table.add_column("VRAM", justify="right")

    providers = ["modal", "runpod", "cloudrun"]
    for p in providers:
        table.add_column(p.upper(), justify="right")
    if show_spot:
        table.add_column("AWS SPOT", justify="right")

    # Collect all GPUs that have at least one offering
    seen_gpus: list[str] = []
    for entry in result.results:
        if entry.gpu not in seen_gpus:
            seen_gpus.append(entry.gpu)

    # Sort by VRAM then by name
    def sort_key(gpu: str):
        spec = GPU_SPECS.get(gpu)
        return (spec.vram_gb if spec else 0, gpu)
    seen_gpus.sort(key=sort_key)

    # Build price lookup: (gpu, provider) -> price
    price_map: dict[tuple[str, str], float] = {}
    for entry in result.results:
        price_map[(entry.gpu, entry.provider)] = entry.price_per_gpu_hour

    for gpu in seen_gpus:
        spec = GPU_SPECS.get(gpu)
        vram = f"{spec.vram_gb} GB" if spec else "?"

        # Collect all prices for this row to find the cheapest
        row_prices: dict[str, float] = {}
        for p in providers:
            price = price_map.get((gpu, p))
            if price is not None and price > 0:
                row_prices[p] = price
        if show_spot:
            sp = spot_prices.get(gpu)
            if sp and sp.price_per_gpu_hour > 0:
                row_prices["aws_spot"] = sp.price_per_gpu_hour

        cheapest_price = min(row_prices.values()) if row_prices else None

        # Build cells, highlighting the cheapest in green
        cells: list[str] = [gpu, vram]
        for p in providers:
            price = price_map.get((gpu, p))
            if price is not None and price > 0:
                text = _format_price(price)
                if price == cheapest_price:
                    cells.append(f"[bold green]{text}[/bold green]")
                else:
                    cells.append(text)
            else:
                cells.append("[dim]-[/dim]")
        if show_spot:
            sp = spot_prices.get(gpu)
            if sp and sp.price_per_gpu_hour > 0:
                text = _format_price(sp.price_per_gpu_hour)
                if sp.price_per_gpu_hour == cheapest_price:
                    cells.append(f"[bold green]{text}[/bold green]")
                else:
                    cells.append(text)
            else:
                cells.append("[dim]-[/dim]")

        table.add_row(*cells)

    console.print(table)


def cmd_list(args: argparse.Namespace) -> None:
    from tandemn.state import list_deployments

    records = list_deployments(status=args.status)
    if not records:
        print("No deployments found.")
        return

    # Print a simple table
    header = f"{'SERVICE NAME':<30} {'STATUS':<12} {'MODEL':<30} {'GPU':<10} {'CREATED':<26}"
    print(header)
    print("-" * len(header))
    for r in records:
        created = r.created_at[:19] if r.created_at else ""
        print(f"{r.service_name:<30} {r.status:<12} {r.model_name:<30} {r.gpu:<10} {created:<26}")


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
    p_deploy.add_argument("--serverless-provider", default=None,
                          help="Serverless backend: modal, runpod, cloudrun (default: cheapest for GPU)")
    p_deploy.add_argument("--spots-cloud", default="aws", help="Cloud for spot GPUs")
    p_deploy.add_argument("--region", default=None)
    p_deploy.add_argument("--concurrency", type=int, default=None,
                          help="Override serverless concurrency limit")
    p_deploy.add_argument("--workers-max", type=int, default=None,
                          help="Max serverless workers (RunPod only)")
    p_deploy.add_argument("--cold-start-mode", default="fast_boot", choices=["fast_boot", "no_fast_boot"])
    p_deploy.add_argument("--no-scale-to-zero", action="store_true")
    p_deploy.add_argument("--scaling-policy", default=None,
                          help="Path to YAML file with scaling parameters")
    p_deploy.add_argument("--service-name", default=None, help="Custom service name")
    p_deploy.add_argument("--public", action="store_true", default=False,
                          help="Make deployed service publicly accessible (no auth). "
                               "WARNING: GPU services are expensive — only use for testing.")
    p_deploy.add_argument("--gcp-project", default=None,
                          help="Google Cloud project ID (overrides GOOGLE_CLOUD_PROJECT env var)")
    p_deploy.add_argument("--gcp-region", default=None,
                          help="Google Cloud region (e.g. us-central1)")
    p_deploy.set_defaults(func=cmd_deploy)

    # -- check --
    p_check = subparsers.add_parser("check", help="Validate provider environment (preflight checks)")
    p_check.add_argument("--provider", required=True, help="Provider to check (e.g. cloudrun)")
    p_check.add_argument("--gpu", default=None, help="GPU type to validate (e.g. L4)")
    p_check.add_argument("--gcp-project", default=None,
                         help="Google Cloud project ID (overrides GOOGLE_CLOUD_PROJECT env var)")
    p_check.add_argument("--gcp-region", default=None,
                         help="Google Cloud region (e.g. us-central1)")
    p_check.set_defaults(func=cmd_check)

    # -- destroy --
    p_destroy = subparsers.add_parser("destroy", help="Tear down a deployment")
    p_destroy.add_argument("--service-name", required=True)
    p_destroy.set_defaults(func=cmd_destroy)

    # -- status --
    p_status = subparsers.add_parser("status", help="Check deployment status")
    p_status.add_argument("--service-name", required=True)
    p_status.set_defaults(func=cmd_status)

    # -- show-gpus --
    p_gpus = subparsers.add_parser("show-gpus", help="Show GPU pricing across providers")
    p_gpus.add_argument("--gpu", default=None, help="Show details for a specific GPU (e.g. L4, H100)")
    p_gpus.add_argument("--provider", default=None, help="Filter to a specific provider")
    p_gpus.add_argument("--spot", action="store_true", default=False,
                        help="Include AWS spot prices (requires SkyPilot)")
    p_gpus.set_defaults(func=cmd_show_gpus)

    # -- list --
    p_list = subparsers.add_parser("list", help="List all deployments")
    p_list.add_argument("--status", default=None, choices=["active", "destroyed", "failed"],
                        help="Filter by deployment status")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
