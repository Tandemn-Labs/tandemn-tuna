"""CLI entry point: python -m tuna deploy|destroy|status|list ..."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from tuna.models import DeployRequest
from tuna.scaling import default_scaling_policy, load_scaling_policy


def cmd_deploy(args: argparse.Namespace) -> None:
    # Lazy import so --help is fast and doesn't pull in heavy deps
    from tuna.orchestrator import launch_hybrid
    from tuna.providers.registry import ensure_provider_registered
    from tuna.state import save_deployment

    _setup_cloud_env(args)

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
        from tuna.catalog import normalize_gpu_name, query as catalog_query
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
        serverless_only=args.serverless_only,
    )

    # Register only the providers we actually need
    try:
        ensure_provider_registered(serverless_provider)
        if not args.serverless_only:
            ensure_provider_registered("skyserve")
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # GCP spot requires skypilot[gcp] for firewall port opening
    if args.spots_cloud == "gcp" and not args.serverless_only:
        try:
            import google.cloud.compute_v1  # noqa: F401 — installed by skypilot[gcp]
        except ImportError:
            print(
                "Error: GCP spot requires additional dependencies.\n"
                "Install with: pip install tandemn-tuna[gcp-spot]",
                file=sys.stderr,
            )
            sys.exit(1)

    # Warn about flags that are ignored in serverless-only mode
    if args.serverless_only:
        ignored = []
        if args.use_different_vm_for_lb:
            ignored.append("--use-different-vm-for-lb")
        if args.no_scale_to_zero:
            ignored.append("--no-scale-to-zero")
        if args.spots_cloud != "aws":
            ignored.append(f"--spots-cloud {args.spots_cloud}")
        if ignored:
            print(f"Warning: {', '.join(ignored)} ignored in serverless-only mode", file=sys.stderr)

    print(f"Deploying {request.model_name} on {request.gpu}")
    print(f"Service name: {request.service_name}")
    if not auto_selected:
        print(f"Serverless provider: {request.serverless_provider}")
    if args.serverless_only:
        print(f"Mode: serverless-only")
    else:
        print(f"Spot cloud: {request.spots_cloud}")
    print()

    from tuna.models import HybridDeployment

    result = None
    try:
        if args.serverless_only:
            from tuna.orchestrator import launch_serverless_only
            result = launch_serverless_only(request)
        else:
            result = launch_hybrid(request, separate_router_vm=args.use_different_vm_for_lb)
    except KeyboardInterrupt:
        print("\nDeployment interrupted! Saving partial state for cleanup...", file=sys.stderr)
    except Exception as e:
        print(f"\nDeployment failed: {e}", file=sys.stderr)
    finally:
        if result is None:
            result = HybridDeployment()
        save_deployment(request, result)

    # Check component outcomes
    serverless_ok = result.serverless and not result.serverless.error
    spot_ok = result.spot and not result.spot.error
    router_ok = result.router and not result.router.error
    has_error = (
        (result.serverless and result.serverless.error)
        or (result.spot and result.spot.error)
        or (result.router and result.router.error)
    )
    total_failure = not (serverless_ok or spot_ok or router_ok)

    if total_failure:
        print("\nDeployment failed: no components launched successfully.", file=sys.stderr)
        print(f"Run: tuna destroy --service-name {request.service_name}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 60)
    print("DEPLOYMENT RESULT")
    print("=" * 60)
    print(f"  vLLM:       {request.vllm_version}")

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
    elif result.spot and result.spot.error:
        print(f"  Spot:       FAILED - {result.spot.error}")
    elif result.spot:
        print(f"  Spot:       launching in background...")

    print()
    if result.router_url:
        if result.router and result.router.endpoint_url:
            print(f"All traffic -> {result.router_url}")
        else:
            # Serverless-only: router_url is the direct serverless endpoint
            print(f"Endpoint -> {result.router_url}")
    print("=" * 60)

    if has_error:
        print()
        print(f"Some components failed. To clean up: tuna destroy --service-name {request.service_name}",
              file=sys.stderr)


def cmd_destroy(args: argparse.Namespace) -> None:
    from tuna.orchestrator import destroy_hybrid
    from tuna.providers.registry import ensure_providers_for_deployment
    from tuna.state import list_deployments, load_deployment, update_deployment_status

    if args.destroy_all:
        from tuna.orchestrator import _cleanup_serve_controller

        records = list_deployments(status="active")
        if not records:
            print("No active deployments to destroy.")
            return
        print(f"Destroying {len(records)} active deployment(s)...")
        errors = []
        for record in records:
            print(f"\n--- {record.service_name} ---")
            try:
                ensure_providers_for_deployment(record)
                destroy_hybrid(record.service_name, record=record,
                               skip_controller_cleanup=True)
                update_deployment_status(record.service_name, "destroyed")
                print(f"Destroyed: {record.service_name}")
            except Exception as e:
                print(f"Failed to destroy {record.service_name}: {e}", file=sys.stderr)
                errors.append(record.service_name)
        # Clean up the SkyServe controller once after all teardowns
        _cleanup_serve_controller()
        if errors:
            print(f"\nFailed to destroy: {', '.join(errors)}", file=sys.stderr)
            sys.exit(1)
        print("\nDone.")
        return

    # Single-service path (unchanged)
    record = load_deployment(args.service_name)
    if record is None:
        print(f"Error: no deployment record found for '{args.service_name}'.", file=sys.stderr)
        sys.exit(1)

    ensure_providers_for_deployment(record)
    print(f"Destroying deployment: {args.service_name}")
    destroy_hybrid(args.service_name, record=record)
    update_deployment_status(args.service_name, "destroyed")

    # Optional: also delete the Azure environment (slow)
    if getattr(args, "azure_cleanup_env", False) and record.serverless_provider_name == "azure":
        from tuna.providers.registry import get_provider
        provider = get_provider("azure")
        if hasattr(provider, "destroy_environment") and record.serverless_metadata:
            from tuna.models import DeploymentResult
            azure_result = DeploymentResult(
                provider="azure",
                metadata=dict(record.serverless_metadata),
            )
            print("Deleting Azure environment (this takes 20+ min)...")
            provider.destroy_environment(azure_result)

    print("Done.")


def cmd_status(args: argparse.Namespace) -> None:
    from tuna.orchestrator import status_hybrid
    from tuna.providers.registry import ensure_providers_for_deployment
    from tuna.state import load_deployment

    record = load_deployment(args.service_name)
    if record is None:
        print(f"Error: no deployment record found for '{args.service_name}'.", file=sys.stderr)
        sys.exit(1)

    ensure_providers_for_deployment(record)

    status = status_hybrid(args.service_name, record=record)
    _print_status(status)


def _print_status(status: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    service_name = status.get("service_name", "?")
    mode = status.get("mode", "")
    mode_suffix = f"  [dim]({mode})[/dim]" if mode else ""
    console.print(f"\n[bold]{service_name}[/bold]{mode_suffix}\n")

    is_serverless_only = mode == "serverless-only"

    # -- Components table --
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Endpoint")

    # Router
    if not is_serverless_only:
        router = status.get("router") or {}
        router_status = router.get("status", "unknown")
        if router.get("url") and router_status != "unreachable":
            router_status = "running"
        router_url = router.get("url", "-")
        _add_status_row(table, "Router", router_status, router_url)

    # Serverless
    sl = status.get("serverless") or {}
    sl_status = sl.get("status", "unknown")
    sl_provider = sl.get("provider", "")
    sl_label = f"Serverless ({sl_provider})" if sl_provider else "Serverless"
    sl_error = sl.get("error")
    sl_endpoint = sl.get("endpoint_url", "-")
    if sl_error:
        sl_status = f"error: {sl_error}"
    _add_status_row(table, sl_label, sl_status, sl_endpoint)

    # Spot
    if not is_serverless_only:
        spot = status.get("spot") or {}
        spot_status = spot.get("status", "unknown")
        spot_provider = spot.get("provider", "")
        spot_label = f"Spot ({spot_provider})" if spot_provider else "Spot"
        spot_endpoint = spot.get("endpoint", "-")
        _add_status_row(table, spot_label, spot_status, spot_endpoint)

    console.print(table)

    # -- Route stats --
    router = status.get("router") or {}
    route_stats = router.get("route_stats")
    if route_stats and route_stats.get("total", 0) > 0:
        console.print()
        stats_table = Table(title="Route stats", show_header=True, header_style="bold")
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("Total requests", str(route_stats.get("total", 0)))
        stats_table.add_row("Serverless", f"{route_stats.get('serverless', 0)} ({route_stats.get('pct_serverless', 0):.0f}%)")
        stats_table.add_row("Spot", f"{route_stats.get('spot', 0)} ({route_stats.get('pct_spot', 0):.0f}%)")
        console.print(stats_table)

    # -- Spot replica details (parse the raw sky output) --
    spot = status.get("spot") or {}
    raw = spot.get("raw", "")
    if raw:
        # Extract just the replica lines for a cleaner view
        lines = raw.strip().splitlines()
        replica_lines = []
        in_replicas = False
        for line in lines:
            if line.startswith("Service Replicas"):
                in_replicas = True
                continue
            if in_replicas and line.strip():
                replica_lines.append(line)

        if replica_lines and len(replica_lines) >= 2:
            console.print()
            # First line is headers, rest are data
            headers = replica_lines[0].split()
            console.print("[bold]Spot replicas[/bold]")
            for line in replica_lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    # Find the status (last word)
                    replica_status = parts[-1] if parts else "?"
                    style = "green" if replica_status == "READY" else "yellow" if replica_status == "STARTING" else "red"
                    console.print(f"  [{style}]{replica_status}[/{style}]  {line.strip()}")

    console.print()


def _add_status_row(table, component: str, status: str, endpoint: str) -> None:
    status_lower = status.lower()
    if status_lower == "running" or status_lower == "ready":
        style = "[bold green]"
        end = "[/bold green]"
    elif "error" in status_lower or "failed" in status_lower or "unreachable" in status_lower:
        style = "[bold red]"
        end = "[/bold red]"
    elif "starting" in status_lower or "not found" in status_lower or "unknown" in status_lower:
        style = "[yellow]"
        end = "[/yellow]"
    else:
        style = ""
        end = ""
    table.add_row(component, f"{style}{status}{end}", endpoint)


def _setup_cloud_env(args: argparse.Namespace) -> None:
    """Forward cloud-related CLI arguments to environment variables."""
    if getattr(args, "gcp_project", None):
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project
    if getattr(args, "gcp_region", None):
        os.environ["GOOGLE_CLOUD_REGION"] = args.gcp_region
    if getattr(args, "azure_subscription", None):
        os.environ["AZURE_SUBSCRIPTION_ID"] = args.azure_subscription
    if getattr(args, "azure_resource_group", None):
        os.environ["AZURE_RESOURCE_GROUP"] = args.azure_resource_group
    if getattr(args, "azure_region", None):
        os.environ["AZURE_REGION"] = args.azure_region
    if getattr(args, "azure_environment", None):
        os.environ["AZURE_ENVIRONMENT"] = args.azure_environment


def cmd_check(args: argparse.Namespace) -> None:
    from tuna.providers.registry import ensure_provider_registered, get_provider

    _setup_cloud_env(args)

    provider_name = args.provider
    try:
        ensure_provider_registered(provider_name)
        provider = get_provider(provider_name)
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Build a minimal DeployRequest for preflight
    default_gpu = "T4" if provider_name in ("azure", "cerebrium") else "L4"
    request = DeployRequest(
        model_name="check",
        gpu=args.gpu or default_gpu,
        region=getattr(args, "gcp_region", None),
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


def cmd_cost(args: argparse.Namespace) -> None:
    from tuna.catalog import (
        fetch_on_demand_prices,
        fetch_spot_prices,
        get_provider_price,
        get_gpu_spec,
    )
    from tuna.orchestrator import status_hybrid
    from tuna.providers.registry import ensure_providers_for_deployment
    from tuna.state import load_deployment

    record = load_deployment(args.service_name)
    if record is None:
        print(f"Error: no deployment record found for '{args.service_name}'.", file=sys.stderr)
        sys.exit(1)

    ensure_providers_for_deployment(record)

    # Detect serverless-only deployment (same heuristic as status_hybrid)
    is_serverless_only = (
        record.serverless_provider_name
        and not record.spot_provider_name
        and not record.router_endpoint
    )

    if is_serverless_only:
        _print_serverless_only_cost(record)
        return

    # Fetch router health for route_stats with cost fields
    status = status_hybrid(args.service_name, record=record)
    router = status.get("router") or {}
    route_stats = router.get("route_stats")

    if route_stats is None:
        print(f"Error: could not reach router for '{args.service_name}'.", file=sys.stderr)
        print("Check deployment status with: tuna status --service-name " + args.service_name, file=sys.stderr)
        sys.exit(1)

    # Look up prices
    serverless_price = get_provider_price(record.gpu, record.serverless_provider)

    spot_prices = fetch_spot_prices(cloud=record.spots_cloud)
    spot_entry = spot_prices.get(record.gpu)
    spot_price = spot_entry.price_per_gpu_hour if spot_entry else 0.0

    on_demand_prices = fetch_on_demand_prices(cloud=record.spots_cloud)
    on_demand_entry = on_demand_prices.get(record.gpu)
    on_demand_price = on_demand_entry.price_per_gpu_hour if on_demand_entry else 0.0

    # Extract cost fields from route_stats
    gpu_sec_svl = route_stats.get("gpu_seconds_serverless", 0.0)
    gpu_sec_spot = route_stats.get("gpu_seconds_spot", 0.0)
    spot_ready_s = route_stats.get("spot_ready_seconds", 0.0)
    uptime_s = route_stats.get("uptime_seconds", 0.0)
    gpu_count = record.gpu_count

    router_meta = record.router_metadata or {}
    ROUTER_CPU_COST_PER_HOUR = 0.0 if router_meta.get("colocated") == "true" else 0.04

    # Actual costs
    actual_serverless = (gpu_sec_svl / 3600) * serverless_price
    actual_spot = (spot_ready_s / 3600) * spot_price * gpu_count
    actual_router = (uptime_s / 3600) * ROUTER_CPU_COST_PER_HOUR
    actual_total = actual_serverless + actual_spot + actual_router

    # Counterfactuals
    all_serverless = ((gpu_sec_svl + gpu_sec_spot) / 3600) * serverless_price
    all_on_demand = (uptime_s / 3600) * on_demand_price * gpu_count

    # Savings
    savings_vs_serverless = all_serverless - actual_total
    savings_vs_on_demand = all_on_demand - actual_total

    _print_cost_dashboard(
        service_name=args.service_name,
        record=record,
        route_stats=route_stats,
        serverless_price=serverless_price,
        spot_price=spot_price,
        spot_entry=spot_entry,
        on_demand_price=on_demand_price,
        on_demand_entry=on_demand_entry,
        gpu_sec_svl=gpu_sec_svl,
        gpu_sec_spot=gpu_sec_spot,
        spot_ready_s=spot_ready_s,
        uptime_s=uptime_s,
        actual_serverless=actual_serverless,
        actual_spot=actual_spot,
        actual_router=actual_router,
        actual_total=actual_total,
        all_serverless=all_serverless,
        all_on_demand=all_on_demand,
        savings_vs_serverless=savings_vs_serverless,
        savings_vs_on_demand=savings_vs_on_demand,
    )


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration (e.g. '2h 34m')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = int(minutes // 60)
    remaining_min = int(minutes % 60)
    return f"{hours}h {remaining_min:02d}m"


def _print_serverless_only_cost(record) -> None:
    from datetime import datetime, timezone
    from rich.console import Console
    from rich.table import Table
    from tuna.catalog import get_provider_price, GPU_SPECS

    console = Console()
    serverless_price = get_provider_price(record.gpu, record.serverless_provider)
    spec = GPU_SPECS.get(record.gpu)
    gpu_label = f"{record.gpu} ({spec.vram_gb} GB)" if spec else record.gpu

    # Compute uptime from created_at
    now = datetime.now(timezone.utc)
    uptime_s = 0.0
    if record.created_at:
        created = datetime.fromisoformat(record.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        uptime_s = (now - created).total_seconds()

    console.print()
    console.print(f"[bold]Cost Dashboard: {record.service_name}[/bold]  [dim](serverless-only)[/dim]")
    console.print(f"GPU: {gpu_label} · Provider: {record.serverless_provider}")
    console.print(f"Uptime: {_format_duration(uptime_s)}")
    console.print()

    # Pricing table
    table = Table(title="Serverless Pricing", show_header=True, header_style="bold", expand=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Provider", record.serverless_provider)
    table.add_row("GPU", gpu_label)
    table.add_row("Rate", f"${serverless_price:.4f}/GPU-hour")
    table.add_row("Deployment uptime", _format_duration(uptime_s))

    max_cost = (uptime_s / 3600) * serverless_price * record.gpu_count
    table.add_row("[bold]Max possible cost[/bold]", f"[bold]${max_cost:.2f}[/bold]")

    console.print(table)
    console.print()
    console.print("[dim]Serverless bills per-second of active compute, not idle time.[/dim]")
    console.print("[dim]Actual cost depends on request volume. Check your provider's billing dashboard.[/dim]")
    console.print()


def _print_cost_dashboard(
    *,
    service_name: str,
    record,
    route_stats: dict,
    serverless_price: float,
    spot_price: float,
    spot_entry,
    on_demand_price: float,
    on_demand_entry,
    gpu_sec_svl: float,
    gpu_sec_spot: float,
    spot_ready_s: float,
    uptime_s: float,
    actual_serverless: float,
    actual_spot: float,
    actual_router: float,
    actual_total: float,
    all_serverless: float,
    all_on_demand: float,
    savings_vs_serverless: float,
    savings_vs_on_demand: float,
) -> None:
    from rich.console import Console
    from rich.table import Table

    from tuna.catalog import GPU_SPECS

    console = Console()

    # Header
    spec = GPU_SPECS.get(record.gpu)
    gpu_label = f"{record.gpu} ({spec.vram_gb} GB)" if spec else record.gpu

    total_reqs = route_stats.get("total", 0)
    pct_spot = route_stats.get("pct_spot", 0.0)
    pct_svl = route_stats.get("pct_serverless", 0.0)

    console.print()
    console.print(f"[bold]Cost Dashboard: {service_name}[/bold]")
    console.print(
        f"GPU: {gpu_label} \u00b7 Serverless: {record.serverless_provider} \u00b7 Spot: {record.spots_cloud}"
    )
    console.print(
        f"Uptime: {_format_duration(uptime_s)} \u00b7 "
        f"{total_reqs:,} requests ({pct_spot:.0f}% spot, {pct_svl:.0f}% serverless)"
    )
    console.print()

    # Actual Costs table
    cost_table = Table(title="Actual Costs", show_header=True, header_style="bold", expand=True)
    cost_table.add_column("Component")
    cost_table.add_column("Cost", justify="right")
    cost_table.add_column("Details", style="dim")

    svl_details = f"{gpu_sec_svl:,.0f} GPU-sec"
    cost_table.add_row(
        f"Serverless ({record.serverless_provider})",
        f"${actual_serverless:.2f}",
        svl_details,
    )

    spot_details = f"{_format_duration(spot_ready_s)} ready"
    if spot_price == 0.0:
        spot_details += " (spot price unavailable)"
    cost_table.add_row(
        f"Spot ({record.spots_cloud})",
        f"${actual_spot:.2f}",
        spot_details,
    )

    cost_table.add_row(
        "Router CPU",
        f"${actual_router:.2f}",
        f"{_format_duration(uptime_s)} uptime",
    )

    cost_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]${actual_total:.2f}[/bold]",
        "",
    )

    console.print(cost_table)
    console.print()

    # Counterfactual table
    cf_table = Table(title="If You Had Used", show_header=True, header_style="bold", expand=True)
    cf_table.add_column("Scenario")
    cf_table.add_column("Cost", justify="right")
    cf_table.add_column("vs actual", style="dim")

    if all_serverless > 0 and actual_total > 0:
        ratio_svl = all_serverless / actual_total
        cf_table.add_row(
            "All serverless",
            f"${all_serverless:.2f}",
            f"{ratio_svl:.1f}x more expensive" if ratio_svl > 1 else f"{ratio_svl:.1f}x",
        )
    else:
        cf_table.add_row("All serverless", f"${all_serverless:.2f}", "-")

    if on_demand_price > 0:
        if all_on_demand > 0 and actual_total > 0:
            ratio_od = all_on_demand / actual_total
            cf_table.add_row(
                "All on-demand",
                f"${all_on_demand:.2f}",
                f"{ratio_od:.1f}x more expensive" if ratio_od > 1 else f"{ratio_od:.1f}x",
            )
        else:
            cf_table.add_row("All on-demand", f"${all_on_demand:.2f}", "-")
    else:
        cf_table.add_row("All on-demand", "-", "on-demand price unavailable (SkyPilot not installed?)")

    console.print(cf_table)
    console.print()

    # Summary line
    if total_reqs == 0:
        console.print("[dim]No requests yet — deployment is fresh.[/dim]")
    elif savings_vs_serverless > 0 and all_serverless > 0:
        pct_savings = (savings_vs_serverless / all_serverless) * 100
        console.print(
            f"[bold green]You saved ${savings_vs_serverless:.2f} vs all-serverless "
            f"({pct_savings:.0f}% cheaper)[/bold green]"
        )
    elif actual_total > 0:
        console.print(
            f"Hybrid cost: ${actual_total:.2f} \u00b7 All-serverless would be: ${all_serverless:.2f}"
        )

    console.print()


def cmd_show_gpus(args: argparse.Namespace) -> None:
    from tuna.catalog import (
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
        spot_prices = fetch_spot_prices(cloud=args.spots_cloud)

    result = query(gpu=gpu_filter, provider=args.provider)
    result.spot_prices = spot_prices

    if gpu_filter:
        _print_gpu_detail(gpu_filter, result, spot_prices, get_gpu_spec, spot_cloud=args.spots_cloud)
    else:
        _print_gpu_table(result, spot_prices, show_spot=args.spot, get_gpu_spec=get_gpu_spec, spot_cloud=args.spots_cloud)


def _print_provider_selection(gpu: str, result, selected_provider: str) -> None:
    """Print a compact pricing table showing which provider was auto-selected."""
    from rich.console import Console
    from rich.table import Table
    from tuna.catalog import get_gpu_spec

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


def _spot_savings_pct(spot_price: float, serverless_prices: list[float]) -> float | None:
    """Calculate percentage savings of spot vs cheapest serverless.

    Returns positive = spot is cheaper, negative = spot is more expensive,
    0 = same price, None = can't compute.
    """
    valid = [p for p in serverless_prices if p > 0]
    if not valid or spot_price <= 0:
        return None
    cheapest = min(valid)
    return (cheapest - spot_price) / cheapest * 100


def _print_gpu_detail(gpu: str, result, spot_prices: dict, get_gpu_spec, spot_cloud: str = "aws") -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    spot_label = f"{spot_cloud} spot"

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
        all_prices.append((spot_label, sp.price_per_gpu_hour))
        table.add_row(spot_label, _format_price(sp.price_per_gpu_hour), extra)

    console.print(table)

    if all_prices:
        cheapest = min(all_prices, key=lambda x: x[1])
        console.print(
            f"\nCheapest: [bold green]{cheapest[0]}[/bold green] "
            f"at [bold green]${cheapest[1]:.2f}/hr[/bold green]"
        )

        # Show savings summary: spot vs cheapest serverless
        serverless_only = [(p, pr) for p, pr in all_prices if p != spot_label]
        spot_entry = next(((p, pr) for p, pr in all_prices if p == spot_label), None)
        if spot_entry and serverless_only:
            cheapest_sl = min(serverless_only, key=lambda x: x[1])
            pct = _spot_savings_pct(spot_entry[1], [pr for _, pr in serverless_only])
            if pct is not None:
                cheapest_sl_details = f"({cheapest_sl[0]} ${cheapest_sl[1]:.2f}/hr)"
                if pct > 0:
                    console.print(
                        f"Spot saves [bold green]{pct:.0f}%[/bold green] vs "
                        f"cheapest serverless {cheapest_sl_details}"
                    )
                elif pct == 0:
                    console.print(
                        f"Spot price is the [dim]same[/dim] as cheapest serverless "
                        f"{cheapest_sl_details}"
                    )
                else:
                    console.print(
                        f"Spot is [red]{-pct:.0f}% more[/red] expensive than cheapest serverless "
                        f"{cheapest_sl_details}"
                    )


def _print_gpu_table(result, spot_prices: dict, show_spot: bool, get_gpu_spec, spot_cloud: str = "aws") -> None:
    from rich.console import Console
    from rich.table import Table
    from tuna.catalog import GPU_SPECS

    spot_label = f"{spot_cloud.upper()} SPOT"

    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("GPU")
    table.add_column("VRAM", justify="right")

    providers = ["modal", "runpod", "cloudrun", "baseten", "azure", "cerebrium"]
    for p in providers:
        table.add_column(p.upper(), justify="right")
    if show_spot:
        table.add_column(spot_label, justify="right")
        table.add_column("SAVINGS", justify="right")

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
                row_prices["spot"] = sp.price_per_gpu_hour

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

            # Savings column: spot vs cheapest serverless
            serverless_prices = [
                price_map.get((gpu, p), 0.0) for p in providers
            ]
            spot_price = sp.price_per_gpu_hour if sp and sp.price_per_gpu_hour > 0 else 0.0
            pct = _spot_savings_pct(spot_price, serverless_prices)
            if pct is not None:
                if pct > 0:
                    cells.append(f"[bold green]{pct:.0f}% cheaper[/bold green]")
                elif pct == 0:
                    cells.append("[dim]same[/dim]")
                else:
                    cells.append(f"[red]{-pct:.0f}% more[/red]")
            else:
                cells.append("[dim]-[/dim]")

        table.add_row(*cells)

    console.print(table)


def cmd_list(args: argparse.Namespace) -> None:
    from tuna.state import list_deployments

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
        prog="tuna",
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
                          help="Serverless backend: modal, runpod, cloudrun, baseten, azure, cerebrium (default: cheapest for GPU)")
    p_deploy.add_argument("--spots-cloud", default="aws", choices=["aws", "gcp", "azure"],
                          help="Cloud for spot GPUs: aws, gcp, azure (default: aws)")
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
    p_deploy.add_argument("--serverless-only", action="store_true", default=False,
                          help="Deploy serverless only — no spot, no router. Returns direct provider endpoint.")
    p_deploy.add_argument("--use-different-vm-for-lb", action="store_true", default=False,
                          help="Launch router on separate VM instead of colocating on SkyServe controller")
    p_deploy.add_argument("--gcp-project", default=None,
                          help="Google Cloud project ID (overrides GOOGLE_CLOUD_PROJECT env var)")
    p_deploy.add_argument("--gcp-region", default=None,
                          help="Google Cloud region (e.g. us-central1)")
    p_deploy.add_argument("--azure-subscription", default=None,
                          help="Azure subscription ID")
    p_deploy.add_argument("--azure-resource-group", default=None,
                          help="Azure resource group name")
    p_deploy.add_argument("--azure-region", default=None,
                          help="Azure region (e.g. eastus)")
    p_deploy.add_argument("--azure-environment", default=None,
                          help="Name of existing Container Apps environment to reuse")
    p_deploy.set_defaults(func=cmd_deploy)

    # -- check --
    p_check = subparsers.add_parser("check", help="Validate provider environment (preflight checks)")
    p_check.add_argument("--provider", required=True, help="Provider to check (e.g. cloudrun)")
    p_check.add_argument("--gpu", default=None, help="GPU type to validate (e.g. L4)")
    p_check.add_argument("--gcp-project", default=None,
                         help="Google Cloud project ID (overrides GOOGLE_CLOUD_PROJECT env var)")
    p_check.add_argument("--gcp-region", default=None,
                         help="Google Cloud region (e.g. us-central1)")
    p_check.add_argument("--azure-subscription", default=None,
                         help="Azure subscription ID")
    p_check.add_argument("--azure-resource-group", default=None,
                         help="Azure resource group name")
    p_check.add_argument("--azure-region", default=None,
                         help="Azure region (e.g. eastus)")
    p_check.add_argument("--azure-environment", default=None,
                         help="Name of existing Container Apps environment to reuse")
    p_check.set_defaults(func=cmd_check)

    # -- destroy --
    p_destroy = subparsers.add_parser("destroy", help="Tear down a deployment")
    destroy_group = p_destroy.add_mutually_exclusive_group(required=True)
    destroy_group.add_argument("--service-name", default=None)
    destroy_group.add_argument("--all", action="store_true", dest="destroy_all",
                               help="Destroy all active deployments")
    p_destroy.add_argument("--azure-cleanup-env", action="store_true", default=False,
                           help="Also delete the Azure Container Apps environment (slow, 20+ min)")
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
                        help="Include spot prices (requires SkyPilot)")
    p_gpus.add_argument("--spots-cloud", default="aws", choices=["aws", "gcp", "azure"],
                        help="Cloud for spot prices: aws, gcp, azure (default: aws)")
    p_gpus.set_defaults(func=cmd_show_gpus)

    # -- cost --
    p_cost = subparsers.add_parser("cost", help="Show cost savings dashboard for a deployment")
    p_cost.add_argument("--service-name", required=True)
    p_cost.set_defaults(func=cmd_cost)

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

    # Silence noisy third-party loggers unless --verbose
    if not args.verbose:
        for name in ("azure", "azure.core", "azure.identity", "alembic"):
            logging.getLogger(name).setLevel(logging.WARNING)

    args.func(args)


if __name__ == "__main__":
    main()
