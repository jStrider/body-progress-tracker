"""
CLI entry point for body-align.

Usage:
    body-align --front front.jpg --back back.jpg --profile side.jpg
    body-align front.jpg back.jpg side.jpg
    body-align --help
"""

import sys
from datetime import date as DateClass
from pathlib import Path

import click

from body_align.align import align_photos


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
@click.version_option(package_name="body-align")
def main(photos, front_opt, back_opt, profile_opt, output, date, bg_color, size, no_bg):
    """
    Align progress photos using MediaPipe pose detection + rembg.

    Provide photos as positional args (front back profile) or via named flags.

    Examples:

    \b
        body-align front.jpg back.jpg side.jpg
        body-align --front front.jpg --back back.jpg --profile side.jpg
        body-align front.jpg back.jpg side.jpg --output ./results --date 2026-03-28
        body-align front.jpg back.jpg side.jpg --no-bg --bg-color "#000000"
    """
    # Resolve photo paths: named flags take priority, then positional args
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
        click.echo("Error: no photos provided. Use --help for usage.", err=True)
        sys.exit(1)

    if date is None:
        date = DateClass.today().isoformat()

    click.echo(f"📸 body-align — processing photos for {date}")
    if front:
        click.echo(f"  front:   {front}")
    if back:
        click.echo(f"  back:    {back}")
    if profile:
        click.echo(f"  profile: {profile}")
    click.echo(f"  output:  {output}/{date}/")
    click.echo(f"  size:    {size}, bg: {bg_color}, remove-bg: {not no_bg}")
    click.echo()

    options = {
        "bg_color": bg_color,
        "size": size,
        "no_bg": no_bg,
    }

    result = align_photos(
        front_path=front or "",
        back_path=back or "",
        profile_path=profile or "",
        output_dir=output,
        date=date,
        options=options,
    )

    if result["success"]:
        click.echo()
        click.echo(f"✅ Done! Aligned photos saved to {output}/{date}/")
        for view in ["front", "back", "profile"]:
            if view in result:
                click.echo(f"   {view}: {result[view]}")
    else:
        click.echo("❌ No photos were processed successfully.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
