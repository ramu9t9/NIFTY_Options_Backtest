"""
NiceGUI v1: Run + Compare backtests.

- Tab 1 Run: load config, override key params, run_backtest(), display metrics + equity + tables.
- Tab 2 Compare: scan runs/*/manifest.json, select run_ids, overlay equity curves, compare metrics + tables.

No duplicated logic: calls index_options_bt.run.runner.run_backtest().
"""

from __future__ import annotations

import json
import io
import sys
import zipfile
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional dependency guard
try:
    from nicegui import ui, run

    HAS_NICEGUI = True
except ImportError:
    HAS_NICEGUI = False

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index_options_bt.config import load_config
from index_options_bt.strategy import discover_strategies, list_strategies
from index_options_bt.run.runner import run_backtest, RunResult


if not HAS_NICEGUI:
    print("=" * 70)
    print("NiceGUI is not installed.")
    print("=" * 70)
    print("Install: pip install nicegui")
    print("Or use CLI: py -m index_options_bt.run --config configs/breakout_nifty_sqlite.yaml")
    print("=" * 70)
    sys.exit(0)


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"
CONFIGS_DIR = REPO_ROOT / "configs"


def _as_posix_rel(path: Path) -> str:
    """Return a stable posix-style relative path from repo root, e.g. configs/x.yaml."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _scan_configs() -> List[str]:
    if not CONFIGS_DIR.exists():
        return []
    out: List[str] = []
    for p in sorted(CONFIGS_DIR.glob("*.y*ml")):
        out.append(_as_posix_rel(p))
    return out


def _scan_runs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not RUNS_ROOT.exists():
        return out
    for mf in sorted(RUNS_ROOT.glob("run-*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            out.append(
                {
                    "run_id": data.get("run_id") or mf.parent.name,
                    "created_at": data.get("created_at", ""),
                    "path": str(mf.parent),
                }
            )
        except Exception:
            continue
    return out


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    fp = run_dir / name
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def _read_json(run_dir: Path, name: str) -> Dict[str, Any]:
    fp = run_dir / name
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _read_text(run_dir: Path, name: str, max_chars: int = 200_000) -> str:
    fp = run_dir / name
    if not fp.exists():
        return ""
    try:
        txt = fp.read_text(encoding="utf-8", errors="replace")
        return txt[-max_chars:] if len(txt) > max_chars else txt
    except Exception:
        return ""


def _zip_run_dir(run_dir: Path) -> bytes:
    """Create a zip of key artifacts for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in [
            "config_resolved.json",
            "config_resolved.yaml",
            "manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "positions.csv",
            "selection.csv",
            "metrics.json",
            "report.png",
            "run.log",
            "nicegui_equity.png",
        ]:
            fp = run_dir / name
            if fp.exists() and fp.is_file():
                zf.write(fp, arcname=name)
    return buf.getvalue()


def _plot_equity(run_dirs: List[Path], out_path: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # Dark mode style
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    
    for rd in run_dirs:
        ec = _read_csv(rd, "equity_curve.csv")
        if ec.empty or "equity" not in ec.columns:
            continue
        ax.plot(ec["equity"].values, label=rd.name, linewidth=2)
    
    ax.set_title("Equity Curve Overlay", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Bar index", color="white")
    ax.set_ylabel("Equity", color="white")
    ax.grid(True, alpha=0.3, color="gray")
    ax.legend(loc="best", fontsize=8, facecolor="#2e2e2e", edgecolor="gray", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_color("gray")
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")
    ax.spines["right"].set_color("gray")
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#1e1e1e", edgecolor="none")
    plt.close(fig)
    return out_path


@ui.page("/")
def page() -> None:
    ui.page_title("Index Options Backtest Engine (NiceGUI)")
    ui.dark_mode().enable()
    discover_strategies()

    with ui.header().classes("items-center"):
        ui.label("Index Options Backtest Engine").classes("text-h5")

    tabs = ui.tabs().classes("w-full")
    tab_run = ui.tab("Run")
    tab_compare = ui.tab("Compare")

    with ui.tab_panels(tabs, value=tab_run).classes("w-full"):
        with ui.tab_panel(tab_run):
            # Use a splitter so the "right side" stays on the right (no responsive wrapping).
            splitter = ui.splitter(value=35).classes("w-full")
            with splitter.before:
                with ui.column().classes("w-full gap-2"):
                    cfg_options = _scan_configs()
                    default_cfg = "configs/breakout_nifty_sqlite.yaml"
                    if cfg_options and default_cfg not in cfg_options:
                        default_cfg = cfg_options[0]
                    if not cfg_options:
                        default_cfg = ""

                    # IMPORTANT: NiceGUI select throws if value is not in options.
                    cfg_select_value = default_cfg if default_cfg in cfg_options else None
                    cfg_select = ui.select(cfg_options, value=cfg_select_value, label="Config file (choose)").classes("w-full")
                    cfg_path = ui.input("Config file (path)", value=default_cfg).classes("w-full")
                    ui.label("Tip: pick from the dropdown to avoid typos.").classes("text-xs text-grey-400 -mt-2")

                    def _sync_cfg_from_select() -> None:
                        if cfg_select.value:
                            cfg_path.value = str(cfg_select.value)

                    def _refresh_configs() -> None:
                        opts = _scan_configs()
                        cfg_select.options = opts
                        # keep value valid
                        if cfg_select.value not in opts:
                            cfg_select.value = opts[0] if opts else None
                        if cfg_path.value and str(cfg_path.value) not in opts and opts:
                            cfg_path.value = opts[0]
                        ui.notify(f"Found {len(opts)} config(s)", type="info")

                    cfg_select.on("update:model-value", lambda _: _sync_cfg_from_select())
                    ui.button("Refresh config list").props("outline dense").classes("w-full").on("click", lambda _: _refresh_configs())

                    strat = ui.select(list_strategies(), value="breakout", label="Strategy").classes("w-full")
                    start = ui.input("Start (YYYY-MM-DD)", value="2025-10-01").classes("w-full")
                    end = ui.input("End (YYYY-MM-DD)", value="2025-10-15").classes("w-full")
                    bar = ui.input("Bar size", value="15s").classes("w-full")

                    ui.separator()
                    ui.label("Contract Selection").classes("text-subtitle2 font-bold")
                    sel_mode = ui.select(["atm", "dte", "delta"], value="atm", label="Selector mode").classes("w-full")

                    contract_multiplier = ui.number(
                        label="Lot size (contract multiplier)",
                        value=75,
                        min=1,
                        max=1000,
                        format="%.0f",
                    ).classes("w-full")
                    ui.label("(NIFTY lot size you want to use for P&L, e.g., 75)").classes("text-xs text-grey-400 -mt-2")
                    
                    target_dte_container = ui.row().classes("w-full items-center gap-2")
                    with target_dte_container:
                        target_dte = ui.number(label="Target DTE", value=None, min=0, max=30, format="%.0f").classes("flex-1")
                        ui.label("(Days To Expiry, e.g., 0 for same-day, 7 for weekly)").classes("text-xs text-grey-400")
                    target_dte_container.set_visibility(False)
                    
                    target_delta_container = ui.row().classes("w-full items-center gap-2")
                    with target_delta_container:
                        target_delta = ui.number(label="Target delta", value=None, min=-1.0, max=1.0, format="%.2f").classes("flex-1")
                        ui.label("(e.g., 0.50 for ATM, 0.30 for OTM, 0.70 for ITM)").classes("text-xs text-grey-400")
                    target_delta_container.set_visibility(False)
                    
                    def _update_selector_visibility():
                        mode = sel_mode.value
                        target_dte_container.set_visibility(mode == "dte")
                        target_delta_container.set_visibility(mode == "delta")
                    
                    sel_mode.on("update:model-value", lambda _: _update_selector_visibility())

                    ui.separator()
                    ui.label("Risk Management (Strategy Parameters)").classes("text-subtitle2 font-bold")
                    take_profit = ui.number("Take Profit (TP) %", value=None, min=0, max=100, format="%.2f").classes("w-full")
                    ui.label("(e.g., 8 = 8% profit target on option premium)").classes("text-xs text-grey-400 -mt-2")
                    stop_loss = ui.number("Stop Loss (SL) %", value=None, min=0, max=100, format="%.2f").classes("w-full")
                    ui.label("(e.g., 5 = 5% loss limit on option premium)").classes("text-xs text-grey-400 -mt-2")

                    ui.separator()
                    ui.label("Execution & Risk").classes("text-subtitle2 font-bold")
                    slippage = ui.number("Slippage (bps)", value=2.0, format="%.1f").classes("w-full")
                    max_contracts = ui.number("Max contracts", value=4, format="%.0f").classes("w-full")

                    run_btn = ui.button("Run backtest", icon="play_arrow").classes("w-full")

            with splitter.after:
                # Right pane - Progress at TOP, results BELOW
                right_column = ui.column().classes("w-full gap-4")
                with right_column:
                    # Progress bar card - FIRST ELEMENT (top of right column)
                    progress_container = ui.card().classes("w-full p-4 bg-blue-900/20 border border-blue-500/30")
                    with progress_container:
                        progress_label = ui.label("Running backtest...").classes("text-lg font-bold mb-2")
                        progress_bar = ui.linear_progress().props("indeterminate").classes("w-full mb-2")
                        progress_status = ui.label("Initializing...").classes("text-sm text-grey-400")
                    progress_container.set_visibility(False)
                    
                    # Results section - SECOND ELEMENT (below progress)
                    results_card = ui.card().classes("w-full p-4")
                    with results_card:
                        results_header = ui.label("Backtest Results").classes("text-h6 font-bold mb-2")
                        metrics_md = ui.markdown("No run yet.").classes("mb-2")
                        run_dir_md = ui.markdown("").classes("mb-2")

                        ui.separator().classes("my-3")
                        ui.label("Artifacts").classes("text-subtitle1 font-bold mb-2")
                        dl_row = ui.row().classes("w-full items-center gap-2")
                        with dl_row:
                            dl_trades_btn = ui.button("Download trades.csv").props("outline dense").classes("flex-1")
                            dl_equity_btn = ui.button("Download equity_curve.csv").props("outline dense").classes("flex-1")
                            dl_selection_btn = ui.button("Download selection.csv").props("outline dense").classes("flex-1")
                            dl_zip_btn = ui.button("Download run bundle (.zip)").props("outline dense").classes("flex-1")

                        ui.separator().classes("my-3")
                        ui.label("Run log").classes("text-subtitle1 font-bold mb-2")
                        log_box = ui.textarea(value="", label="run.log (tail)").props("readonly").classes("w-full")
                        log_refresh_btn = ui.button("Refresh log").props("outline dense").classes("w-full")
                        eq_img = ui.image("").classes("w-full mb-4").set_visibility(False)
                        
                        ui.separator().classes("my-4")
                        trades_label = ui.label("Trades").classes("text-subtitle1 font-bold mb-2")
                        trades_tbl = ui.table(columns=[], rows=[]).classes("w-full mb-4")
                        
                        ui.separator().classes("my-4")
                        selection_label = ui.label("Contract Selections").classes("text-subtitle1 font-bold mb-2")
                        sel_tbl = ui.table(columns=[], rows=[]).classes("w-full")
                    
                    # Initially hide results elements
                    results_header.set_visibility(False)
                    dl_row.set_visibility(False)
                    dl_trades_btn.set_visibility(False)
                    dl_equity_btn.set_visibility(False)
                    dl_selection_btn.set_visibility(False)
                    dl_zip_btn.set_visibility(False)
                    log_box.set_visibility(False)
                    log_refresh_btn.set_visibility(False)
                    trades_label.set_visibility(False)
                    trades_tbl.set_visibility(False)
                    selection_label.set_visibility(False)
                    sel_tbl.set_visibility(False)

            async def _run() -> None:
                # NOTE: This function is intentionally async to avoid blocking the NiceGUI event loop.
                # If we block the event loop during a long backtest, the browser websocket disconnects ("Connection lost").
                try:
                    # Show progress (right side), keep results container visible so we can update text live
                    run_btn.disable()
                    if progress_container is not None:
                        progress_container.set_visibility(True)
                    if results_header is not None:
                        results_header.set_visibility(True)
                    if eq_img is not None:
                        eq_img.set_visibility(False)
                    if trades_label is not None:
                        trades_label.set_visibility(False)
                    if trades_tbl is not None:
                        trades_tbl.set_visibility(False)
                    if selection_label is not None:
                        selection_label.set_visibility(False)
                    if sel_tbl is not None:
                        sel_tbl.set_visibility(False)
                    if progress_label is not None:
                        progress_label.text = "Running backtest..."
                    if progress_status is not None:
                        progress_status.text = "Loading configuration..."
                    ui.notify("Backtest started…", type="info")
                    if metrics_md is not None:
                        metrics_md.content = "**Status**: Running backtest…"
                    if run_dir_md is not None:
                        run_dir_md.content = ""

                    cfg_file = str(cfg_path.value or "").strip()
                    if not cfg_file:
                        raise FileNotFoundError("Config file path is empty.")

                    # Resolve relative paths against repo root (so running from apps/ works)
                    cfg_path_obj = Path(cfg_file)
                    if not cfg_path_obj.is_absolute():
                        cfg_path_obj = (REPO_ROOT / cfg_path_obj).resolve()

                    if not cfg_path_obj.exists():
                        opts = _scan_configs()
                        suggestion = ""
                        if opts:
                            matches = difflib.get_close_matches(cfg_file, opts, n=1, cutoff=0.3)
                            if matches:
                                suggestion = matches[0]
                        msg = f"Config file not found: {cfg_file}"
                        if suggestion:
                            msg += f"\n\nDid you mean: {suggestion}"
                            # Auto-fill the closest match to help the user recover quickly
                            cfg_path.value = suggestion
                            cfg_select.value = suggestion
                        msg += f"\n\nAvailable configs:\n- " + "\n- ".join(opts) if opts else "\n\nNo configs found under ./configs/"
                        raise FileNotFoundError(msg)

                    cfg = load_config(str(cfg_path_obj))
                    cfg.engine.start = start.value
                    cfg.engine.end = end.value
                    cfg.engine.bar_size = bar.value
                    cfg.strategy.name = strat.value

                    if progress_status is not None:
                        progress_status.text = "Updating strategy parameters..."

                    # Update strategy params with TP/SL if provided
                    if take_profit.value is not None:
                        if cfg.strategy.params is None:
                            cfg.strategy.params = {}
                        cfg.strategy.params["take_profit_pct"] = float(take_profit.value)
                    if stop_loss.value is not None:
                        if cfg.strategy.params is None:
                            cfg.strategy.params = {}
                        cfg.strategy.params["stop_loss_pct"] = float(stop_loss.value)

                    cfg.selector.mode = sel_mode.value
                    cfg.selector.target_dte = int(target_dte.value) if target_dte.value is not None else None
                    cfg.selector.target_delta = float(target_delta.value) if target_delta.value is not None else None
                    cfg.selector.contract_multiplier = int(contract_multiplier.value) if contract_multiplier.value is not None else 75

                    cfg.execution.slippage_bps = float(slippage.value)
                    cfg.risk.max_contracts = int(max_contracts.value)

                    if progress_status is not None:
                        progress_status.text = "Executing backtest (this may take a while)..."
                    # Run in a worker thread to keep UI responsive
                    res: RunResult = await run.io_bound(run_backtest, cfg, "timestamp")

                    if progress_status is not None:
                        progress_status.text = "Generating results..."
                    await run.io_bound(lambda: None)  # Small delay to update UI

                    # Hide progress, show results
                    if progress_container is not None:
                        progress_container.set_visibility(False)
                    
                    # Ensure results header and content are visible
                    if results_header is not None:
                        results_header.set_visibility(True)
                    # Make sure results card is visible (it should be, but ensure it)
                    if results_card is not None:
                        results_card.set_visibility(True)

                    m = res.metrics
                    if metrics_md is not None:
                        metrics_md.content = (
                            f"**Run ID**: `{res.run_id}`\n\n"
                            f"### Performance Metrics\n\n"
                            f"- **Total Return**: {m.get('total_return_pct', 0.0):.2f}%\n"
                            f"- **Max Drawdown**: {m.get('max_drawdown_pct', 0.0):.2f}%\n"
                            f"- **Sharpe Ratio**: {m.get('sharpe', 0.0):.2f}\n"
                            f"- **Total Trades**: {m.get('total_trades', 0)}\n"
                            f"- **Win Rate**: {m.get('win_rate_pct', 0.0):.2f}%\n"
                            f"- **Avg Trade P&L**: ₹{m.get('avg_trade_pnl', 0.0):,.2f}\n"
                            f"- **Final Equity**: ₹{m.get('final_equity', 0.0):,.2f}\n"
                        )
                    if run_dir_md is not None:
                        run_dir_md.content = f"**Run Directory**: `{res.run_dir}`"

                    # Enable downloads + log viewer
                    dl_row.set_visibility(True)
                    dl_trades_btn.set_visibility(True)
                    dl_equity_btn.set_visibility(True)
                    dl_selection_btn.set_visibility(True)
                    dl_zip_btn.set_visibility(True)
                    log_box.set_visibility(True)
                    log_refresh_btn.set_visibility(True)

                    def _download_file(name: str) -> None:
                        fp = res.run_dir / name
                        if not fp.exists():
                            ui.notify(f"Missing artifact: {name}", type="warning")
                            return
                        ui.download(str(fp), filename=name)

                    def _download_zip() -> None:
                        data = _zip_run_dir(res.run_dir)
                        ui.download(data, filename=f"{res.run_id}.zip")

                    def _refresh_log() -> None:
                        log_box.value = _read_text(res.run_dir, "run.log")

                    # Bind buttons for this run (each run rebinds to latest run_dir)
                    dl_trades_btn.on("click", lambda _: _download_file("trades.csv"))
                    dl_equity_btn.on("click", lambda _: _download_file("equity_curve.csv"))
                    dl_selection_btn.on("click", lambda _: _download_file("selection.csv"))
                    dl_zip_btn.on("click", lambda _: _download_zip())
                    log_refresh_btn.on("click", lambda _: _refresh_log())
                    _refresh_log()

                    # equity image
                    try:
                        p = _plot_equity([res.run_dir], res.run_dir / "nicegui_equity.png")
                        if p is not None and p.exists() and eq_img is not None:
                            eq_img.source = str(p)
                            eq_img.set_visibility(True)
                    except Exception as e:
                        if progress_status is not None:
                            progress_status.text = f"Warning: Could not generate equity plot: {e}"

                    # trades table
                    tdf = _read_csv(res.run_dir, "trades.csv").fillna("")
                    if not tdf.empty and trades_tbl is not None and trades_label is not None:
                        trades_tbl.columns = [{"name": c, "label": c, "field": c} for c in tdf.columns]
                        trades_tbl.rows = tdf.to_dict("records")
                        trades_label.set_visibility(True)
                        trades_tbl.set_visibility(True)
                    else:
                        if trades_label is not None:
                            trades_label.set_visibility(False)
                        if trades_tbl is not None:
                            trades_tbl.set_visibility(False)

                    # selection table
                    sdf = _read_csv(res.run_dir, "selection.csv").fillna("")
                    # cap for UI responsiveness
                    sdf = sdf.head(500)
                    if not sdf.empty and sel_tbl is not None and selection_label is not None:
                        sel_tbl.columns = [{"name": c, "label": c, "field": c} for c in sdf.columns]
                        sel_tbl.rows = sdf.to_dict("records")
                        selection_label.set_visibility(True)
                        sel_tbl.set_visibility(True)
                    else:
                        if selection_label is not None:
                            selection_label.set_visibility(False)
                        if sel_tbl is not None:
                            sel_tbl.set_visibility(False)

                    ui.notify(f"Backtest completed: {m.get('total_trades', 0)} trades, {m.get('total_return_pct', 0.0):.2f}% return", type="positive")
                except Exception as e:
                    import traceback
                    error_msg = f"**Error**: {str(e)}\n\n```\n{traceback.format_exc()}\n```\n\nPlease check your configuration and try again."
                    if progress_container is not None:
                        progress_container.set_visibility(False)
                    if results_header is not None:
                        results_header.set_visibility(True)
                    if results_card is not None:
                        results_card.set_visibility(True)
                    if metrics_md is not None:
                        metrics_md.content = error_msg
                    ui.notify(f"Run failed: {e}", type="negative")
                finally:
                    run_btn.enable()

            run_btn.on("click", _run)

        with ui.tab_panel(tab_compare):
            with ui.row().classes("w-full"):
                with ui.column().classes("w-1/3 gap-2"):
                    runs = _scan_runs()
                    run_ids = [r["run_id"] for r in runs]
                    run_select = ui.select(run_ids, label="Run IDs", multiple=True).classes("w-full")
                    refresh_btn = ui.button("Refresh").classes("w-full")
                    compare_btn = ui.button("Compare").classes("w-full")

                with ui.column().classes("w-2/3 gap-2"):
                    overlay_img = ui.image("").classes("w-full").set_visibility(False)
                    metrics_tbl = ui.table(columns=[], rows=[]).classes("w-full")
                    trades_tbl2 = ui.table(columns=[], rows=[]).classes("w-full")
                    sel_tbl2 = ui.table(columns=[], rows=[]).classes("w-full")
                    compare_dl_row = ui.row().classes("w-full items-center gap-2")
                    with compare_dl_row:
                        dl_overlay_btn = ui.button("Download overlay.png").props("outline dense").classes("flex-1")
                        dl_metrics_btn = ui.button("Download metrics.csv").props("outline dense").classes("flex-1")
                        dl_trades_btn2 = ui.button("Download trades.csv").props("outline dense").classes("flex-1")
                        dl_sel_btn2 = ui.button("Download selection.csv").props("outline dense").classes("flex-1")
                    compare_dl_row.set_visibility(False)

            def _refresh() -> None:
                rs = _scan_runs()
                run_select.options = [r["run_id"] for r in rs]
                ui.notify(f"Found {len(rs)} runs", type="info")

            def _compare() -> None:
                selected = run_select.value or []
                if not selected:
                    ui.notify("Select at least 1 run", type="warning")
                    return
                rdirs = [RUNS_ROOT / rid for rid in selected if (RUNS_ROOT / rid).exists()]
                if not rdirs:
                    ui.notify("No run dirs found for selection", type="warning")
                    return

                p = _plot_equity(rdirs, RUNS_ROOT / "_compare" / "overlay.png")
                if p is not None and p.exists():
                    overlay_img.source = str(p)
                    overlay_img.set_visibility(True)

                # metrics
                mrows = []
                for rd in rdirs:
                    m = _read_json(rd, "metrics.json")
                    mrows.append({"run_id": rd.name, **m})
                mdf = pd.DataFrame(mrows).fillna("")
                metrics_tbl.columns = [{"name": c, "label": c, "field": c} for c in mdf.columns] if not mdf.empty else []
                metrics_tbl.rows = mdf.to_dict("records") if not mdf.empty else []

                # concat trades/selection (cap)
                tdfs = []
                sdfs = []
                for rd in rdirs:
                    t = _read_csv(rd, "trades.csv")
                    if not t.empty:
                        t["run_id"] = rd.name
                        tdfs.append(t)
                    s = _read_csv(rd, "selection.csv")
                    if not s.empty:
                        s["run_id"] = rd.name
                        sdfs.append(s)
                if tdfs:
                    tcat = pd.concat(tdfs, ignore_index=True).fillna("").head(500)
                    trades_tbl2.columns = [{"name": c, "label": c, "field": c} for c in tcat.columns]
                    trades_tbl2.rows = tcat.to_dict("records")
                if sdfs:
                    scat = pd.concat(sdfs, ignore_index=True).fillna("").head(500)
                    sel_tbl2.columns = [{"name": c, "label": c, "field": c} for c in scat.columns]
                    sel_tbl2.rows = scat.to_dict("records")

                # Write compare outputs + enable downloads
                out_dir = RUNS_ROOT / "_compare"
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    mdf.to_csv(out_dir / "metrics.csv", index=False)
                    if tdfs:
                        pd.concat(tdfs, ignore_index=True).fillna("").to_csv(out_dir / "trades.csv", index=False)
                    else:
                        pd.DataFrame().to_csv(out_dir / "trades.csv", index=False)
                    if sdfs:
                        pd.concat(sdfs, ignore_index=True).fillna("").to_csv(out_dir / "selection.csv", index=False)
                    else:
                        pd.DataFrame().to_csv(out_dir / "selection.csv", index=False)
                except Exception:
                    pass

                compare_dl_row.set_visibility(True)
                dl_overlay_btn.on("click", lambda _: ui.download(str(out_dir / "overlay.png"), filename="overlay.png"))
                dl_metrics_btn.on("click", lambda _: ui.download(str(out_dir / "metrics.csv"), filename="metrics.csv"))
                dl_trades_btn2.on("click", lambda _: ui.download(str(out_dir / "trades.csv"), filename="trades.csv"))
                dl_sel_btn2.on("click", lambda _: ui.download(str(out_dir / "selection.csv"), filename="selection.csv"))

                ui.notify("Compare complete", type="positive")

            refresh_btn.on("click", lambda e: _refresh())
            compare_btn.on("click", lambda e: _compare())


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8080, title="Index Options Backtest Engine", reload=False)


