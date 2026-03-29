"""
CLI entry point for body-align.

Usage:
    body-align --front front.jpg --back back.jpg --profile side.jpg
    body-align front.jpg back.jpg side.jpg
    body-align --batch ./photos/
    body-align compare 2026-03-01 2026-03-28
    body-align timelapse ./aligned/ --views front --fps 2
    body-align verify ./aligned/
    body-align --help
"""

import json
import sys
from datetime import date as DateClass
from pathlib import Path
from typing import Optional

import click

from body_align.align import align_photos


def _load_config(config_file: Optional[str]) -> dict:
    """Load options from YAML config file."""
    if config_file is None:
        return {}
    try:
        import yaml
    except ImportError:
        click.echo("Error: pyyaml is required for --config. Run: pip install pyyaml", err=True)
        sys.exit(1)
    with open(config_file) as f:
        return yaml.safe_load(f) or {}


@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@click.argument("photos", nargs=-1, metavar="[FRONT BACK PROFILE]")
@click.option("--front", "front_opt", type=click.Path(), default=None, help="Front photo path")
@click.option("--back", "back_opt", type=click.Path(), default=None, help="Back photo path")
@click.option(
    "--profile",
    "profile_opt",
    type=click.Path(),
    default=None,
    help="Profile/side photo path",
)
@click.option(
    "--output",
    "-o",
    default="./aligned",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--date",
    default=None,
    help="Date label for output subdirectory (default: today, YYYY-MM-DD)",
)
@click.option(
    "--bg-color",
    default="#1E1E1E",
    show_default=True,
    help="Background color as hex (#RRGGBB)",
)
@click.option(
    "--size",
    default="800x1067",
    show_default=True,
    help="Output image size as WxH",
)
@click.option(
    "--no-bg",
    is_flag=True,
    default=False,
    help="Skip background removal (faster, keeps original background)",
)
# Batch options
@click.option(
    "--batch",
    "batch_dir",
    type=click.Path(),
    default=None,
    help="Process whole directory (DIR/YYYY-MM-DD/front.jpg structure)",
)
@click.option(
    "--since",
    default=None,
    help="With --batch: only process dates >= this date (YYYY-MM-DD)",
)
# Extended align options
@click.option(
    "--views",
    default=None,
    help="Comma-separated views to process: front,back,profile",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["jpg", "png"], case_sensitive=False),
    default="jpg",
    show_default=True,
    help="Output image format",
)
@click.option(
    "--quality",
    type=click.IntRange(1, 100),
    default=92,
    show_default=True,
    help="JPEG quality (1-100)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without writing files",
)
@click.option(
    "--no-force-align",
    is_flag=True,
    default=False,
    help="Skip the segmentation second pass (faster, less precise)",
)
@click.option(
    "--pose-model",
    type=click.Choice(["lite", "full", "heavy"], case_sensitive=False),
    default="full",
    show_default=True,
    help="MediaPipe pose model complexity",
)
@click.option(
    "--stats",
    is_flag=True,
    default=False,
    help="Print alignment metrics (rotation angle, scale factor, head_y, feet_y)",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON result",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Load options from YAML config file",
)
@click.version_option(package_name="body-align")
def main(
    ctx,
    photos,
    front_opt,
    back_opt,
    profile_opt,
    output,
    date,
    bg_color,
    size,
    no_bg,
    batch_dir,
    since,
    views,
    output_format,
    quality,
    overwrite,
    dry_run,
    no_force_align,
    pose_model,
    stats,
    json_output,
    config_file,
):
    """
    Align progress photos using MediaPipe pose detection + rembg.

    Provide photos as positional args (front back profile) or via named flags.
    Use --batch DIR to process a whole directory at once.

    Examples:

    \b
        body-align front.jpg back.jpg side.jpg
        body-align --front front.jpg --back back.jpg --profile side.jpg
        body-align --batch ./photos/ --since 2026-03-01
        body-align --batch ./photos/ --dry-run
        body-align --batch ./photos/ --views front,back
        body-align compare 2026-03-01 2026-03-28 --views front
        body-align timelapse ./aligned/ --fps 2 --output timelapse.mp4
        body-align verify ./aligned/
    """
    # If a subcommand was invoked, do nothing here
    if ctx.invoked_subcommand is not None:
        return

    # Load config file if provided
    cfg = _load_config(config_file)

    # Config file values as defaults (CLI flags override)
    if not bg_color and "bg_color" in cfg:
        bg_color = cfg["bg_color"]
    if not size and "size" in cfg:
        size = cfg["size"]

    # Parse views
    view_list = None
    if views:
        view_list = [v.strip() for v in views.split(",")]
    elif "views" in cfg:
        view_list = cfg["views"] if isinstance(cfg["views"], list) else [v.strip() for v in cfg["views"].split(",")]

    # --- Batch mode ---
    if batch_dir:
        _run_batch(
            batch_dir=batch_dir,
            since=since,
            output=output,
            view_list=view_list,
            bg_color=bg_color,
            size=size,
            no_bg=no_bg,
            output_format=output_format,
            quality=quality,
            overwrite=overwrite,
            dry_run=dry_run,
            no_force_align=no_force_align,
            pose_model=pose_model,
            stats=stats,
            json_output=json_output,
        )
        return

    # --- Single-shot mode ---
    front = front_opt
    back = back_opt
    profile = profile_opt

    if photos:
        if len(photos) == 3:
            if front is None:
                front = photos[0]
            if back is None:
                back = photos[1]
            if profile is None:
                profile = photos[2]
        elif len(photos) == 1 and front is None:
            front = photos[0]
        elif len(photos) > 0:
            click.echo(
                f"Error: expected 1 or 3 positional arguments (front [back profile]), got {len(photos)}",
                err=True,
            )
            sys.exit(1)

    if not any([front, back, profile]):
        click.echo(ctx.get_help())
        sys.exit(0)

    if date is None:
        date = DateClass.today().isoformat()

    # Apply view filter
    if view_list:
        if "front" not in view_list:
            front = None
        if "back" not in view_list:
            back = None
        if "profile" not in view_list:
            profile = None

    if dry_run:
        click.echo(f"[dry-run] Would process photos for {date}")
        if front:
            click.echo(f"  front:   {front}")
        if back:
            click.echo(f"  back:    {back}")
        if profile:
            click.echo(f"  profile: {profile}")
        click.echo(f"  output:  {output}/{date}/")
        click.echo(f"  size:    {size}, bg: {bg_color}, format: {output_format}, quality: {quality}")
        return

    click.echo(f"📸 body-align — processing photos for {date}")
    if front:
        click.echo(f"  front:   {front}")
    if back:
        click.echo(f"  back:    {back}")
    if profile:
        click.echo(f"  profile: {profile}")
    click.echo(f"  output:  {output}/{date}/")
    click.echo(f"  size:    {size}, bg: {bg_color}, format: {output_format}, quality: {quality}")
    click.echo()

    options = {
        "bg_color": bg_color,
        "size": size,
        "no_bg": no_bg,
        "quality": quality,
        "output_format": output_format,
        "overwrite": overwrite,
        "no_force_align": no_force_align,
        "pose_model": pose_model,
        "print_stats": stats,
    }

    result = align_photos(
        front_path=front or "",
        back_path=back or "",
        profile_path=profile or "",
        output_dir=output,
        date=date,
        options=options,
    )

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return

    if result["success"]:
        click.echo()
        click.echo(f"✅ Done! Aligned photos saved to {output}/{date}/")
        for view in ["front", "back", "profile"]:
            if view in result:
                click.echo(f"   {view}: {result[view]}")
        if stats and "stats" in result:
            click.echo("\n📊 Alignment stats:")
            for view, s in result["stats"].items():
                click.echo(f"  {view}: {s}")
    else:
        click.echo("❌ No photos were processed successfully.", err=True)
        sys.exit(1)


def _run_batch(
    batch_dir,
    since,
    output,
    view_list,
    bg_color,
    size,
    no_bg,
    output_format,
    quality,
    overwrite,
    dry_run,
    no_force_align,
    pose_model,
    stats,
    json_output,
):
    """Process a whole directory of date-organised photos."""
    import re

    base = Path(batch_dir)
    if not base.exists():
        click.echo(f"Error: batch directory {batch_dir} does not exist", err=True)
        sys.exit(1)

    date_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)
    ])

    if since:
        date_dirs = [d for d in date_dirs if d.name >= since]

    if not date_dirs:
        click.echo(f"No YYYY-MM-DD directories found in {batch_dir}" + (f" since {since}" if since else ""))
        return

    click.echo(f"📦 batch mode: {len(date_dirs)} date(s) to process")
    if since:
        click.echo(f"   since: {since}")
    click.echo()

    all_results = {}
    views_to_process = view_list or ["front", "back", "profile"]

    options = {
        "bg_color": bg_color,
        "size": size,
        "no_bg": no_bg,
        "quality": quality,
        "output_format": output_format,
        "overwrite": overwrite,
        "no_force_align": no_force_align,
        "pose_model": pose_model,
        "print_stats": stats,
    }

    for date_dir in date_dirs:
        date = date_dir.name
        front = back = profile = None

        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            if "front" in views_to_process:
                p = date_dir / f"front.{ext}"
                if p.exists():
                    front = str(p)
                    break
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            if "back" in views_to_process:
                p = date_dir / f"back.{ext}"
                if p.exists():
                    back = str(p)
                    break
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            if "profile" in views_to_process:
                p = date_dir / f"profile.{ext}"
                if p.exists():
                    profile = str(p)
                    break

        if not any([front, back, profile]):
            click.echo(f"  [{date}] no matching photos found, skipping")
            continue

        # Check if already processed (skip if not overwrite)
        out_base = Path(output) / date
        if not overwrite and out_base.exists():
            already_done = all(
                (out_base / f"{v}.jpg").exists()
                for v in views_to_process
                if (front if v == "front" else back if v == "back" else profile)
            )
            if already_done:
                click.echo(f"  [{date}] already aligned, skipping (use --overwrite to redo)")
                continue

        if dry_run:
            click.echo(f"  [{date}] would process: " + ", ".join(filter(None, [
                f"front={front}" if front else None,
                f"back={back}" if back else None,
                f"profile={profile}" if profile else None,
            ])))
            continue

        click.echo(f"  [{date}] processing...")
        result = align_photos(
            front_path=front or "",
            back_path=back or "",
            profile_path=profile or "",
            output_dir=output,
            date=date,
            options=options,
        )
        all_results[date] = result

        if result["success"]:
            saved = [k for k in ["front", "back", "profile"] if k in result]
            click.echo(f"  [{date}] ✅ {', '.join(saved)}")
        else:
            click.echo(f"  [{date}] ❌ failed")

    if json_output:
        click.echo(json.dumps(all_results, indent=2))
        return

    total = len([r for r in all_results.values() if r.get("success")])
    click.echo(f"\n✅ Batch done: {total}/{len(all_results)} dates processed successfully")


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------

@main.command("compare")
@click.argument("date1")
@click.argument("date2")
@click.option(
    "--aligned-dir",
    default="./aligned",
    show_default=True,
    help="Directory with aligned photos (YYYY-MM-DD/view.jpg structure)",
)
@click.option(
    "--views",
    default="front",
    show_default=True,
    help="Comma-separated views to compare",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: <aligned-dir>/comparison_DATE1_DATE2.jpg)",
)
def compare_cmd(date1, date2, aligned_dir, views, output):
    """
    Generate a side-by-side comparison image.

    \b
        body-align compare 2026-03-01 2026-03-28
        body-align compare 2026-03-01 2026-03-28 --views front,back --output compare.jpg
    """
    from body_align.compare import make_comparison

    view_list = [v.strip() for v in views.split(",")]
    click.echo(f"🖼️  Comparing {date1} ↔ {date2} (views: {', '.join(view_list)})")

    try:
        out = make_comparison(
            date1=date1,
            date2=date2,
            aligned_dir=aligned_dir,
            views=view_list,
            output_path=output,
        )
        click.echo(f"✅ Saved to {out}")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# timelapse subcommand
# ---------------------------------------------------------------------------

@main.command("timelapse")
@click.argument("aligned_dir")
@click.option(
    "--views",
    default="front",
    show_default=True,
    help="Comma-separated views to include",
)
@click.option(
    "--fps",
    type=float,
    default=2.0,
    show_default=True,
    help="Frames per second",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output path (default: <aligned-dir>/timelapse_<views>.mp4)",
)
@click.option(
    "--gif",
    "as_gif",
    is_flag=True,
    default=False,
    help="Output GIF instead of MP4",
)
def timelapse_cmd(aligned_dir, views, fps, output, as_gif):
    """
    Generate a timelapse video or GIF from aligned photos.

    \b
        body-align timelapse ./aligned/ --views front --fps 2 --output timelapse.mp4
        body-align timelapse ./aligned/ --views front --gif --output timelapse.gif
    """
    from body_align.timelapse import make_timelapse

    view_list = [v.strip() for v in views.split(",")]
    fmt = "GIF" if as_gif else "MP4"
    click.echo(f"🎬 Generating {fmt} timelapse (views: {', '.join(view_list)}, fps: {fps})")

    try:
        out = make_timelapse(
            aligned_dir=aligned_dir,
            views=view_list,
            fps=fps,
            output_path=output,
            as_gif=as_gif,
        )
        click.echo(f"✅ Saved to {out}")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------

@main.command("verify")
@click.argument("aligned_dir")
@click.option(
    "--views",
    default="front,back,profile",
    show_default=True,
    help="Comma-separated views to verify",
)
@click.option(
    "--threshold",
    type=float,
    default=20.0,
    show_default=True,
    help="Warn if position variance exceeds this many pixels",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output machine-readable JSON",
)
def verify_cmd(aligned_dir, views, threshold, json_output):
    """
    Check alignment consistency across all aligned photos.

    Re-detects poses and reports head/feet position variance.
    Warns if variance > threshold (default 20px).

    \b
        body-align verify ./aligned/
        body-align verify ./aligned/ --views front --threshold 15
    """
    from body_align.verify import verify_alignment

    view_list = [v.strip() for v in views.split(",")]
    click.echo(f"🔍 Verifying alignment consistency (views: {', '.join(view_list)})")
    click.echo()

    results = verify_alignment(
        aligned_dir=aligned_dir,
        views=view_list,
        threshold=threshold,
    )

    if json_output:
        click.echo(json.dumps(results, indent=2))
        return

    if "error" in results:
        click.echo(f"❌ {results['error']}", err=True)
        sys.exit(1)

    all_ok = True
    for view, data in results.items():
        click.echo(f"📷 {view}:")
        if "error" in data:
            click.echo(f"   ❌ {data['error']}")
            all_ok = False
            continue

        click.echo(f"   Photos analyzed: {data['photos']}")
        if data["photos"] > 0:
            hy = data["head_y"]
            fy = data["feet_y"]
            click.echo(f"   Head Y:  range={hy['range']:.1f}px  stdev={hy['stdev']:.1f}px  [{hy['min']:.0f}–{hy['max']:.0f}]")
            click.echo(f"   Feet Y:  range={fy['range']:.1f}px  stdev={fy['stdev']:.1f}px  [{fy['min']:.0f}–{fy['max']:.0f}]")

        if data.get("warnings"):
            for w in data["warnings"]:
                click.echo(f"   ⚠️  {w}")
            all_ok = False
        elif data["photos"] > 0:
            click.echo(f"   ✅ OK")

        if data.get("errors"):
            for e in data["errors"]:
                click.echo(f"   ⚠️  {e}")
        click.echo()

    if all_ok:
        click.echo("✅ All views within alignment threshold")
    else:
        click.echo("⚠️  Some views have alignment issues — consider re-running align")
        sys.exit(1)


if __name__ == "__main__":
    main()
