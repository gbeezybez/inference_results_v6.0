
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import argparse


def iter_subdirs(root: Path, recursive: bool = True) -> Iterable[Path]:
    """
    Yield subdirectories under root.
    - recursive=True  -> all nested subdirectories
    - recursive=False -> only direct children
    """
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_dir())
    else:
        yield from (p for p in root.iterdir() if p.is_dir())


def rename_in_each_subdir(
    source_dir: str | Path,
    src_name: str = "filename1.txt",
    dst_name: str = "filename2.txt",
    recursive: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, list]:
    """
    For each subdirectory under source_dir, if 'src_name' exists in that directory,
    rename it to 'dst_name' within the same directory.

    Returns a dict with keys: 'renamed', 'skipped', 'conflicts'
    - renamed: list of (old_path, new_path)
    - skipped: list of (dir_path, reason)
    - conflicts: list of (old_path, new_path) where dst exists and overwrite=False
    """
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {root}")

    renamed = []
    skipped = []
    conflicts = []

    for d in iter_subdirs(root, recursive=recursive):
        src_path = d / src_name
        if not src_path.exists():
            continue  # no match in this subdir

        if not src_path.is_file():
            skipped.append((d, f"Found {src_name} but it is not a regular file"))
            continue

        dst_path = d / dst_name

        if dst_path.exists():
            if overwrite:
                if dst_path.is_dir():
                    skipped.append((d, f"Destination {dst_name} exists and is a directory"))
                    continue
                if not dry_run:
                    dst_path.unlink()
            else:
                conflicts.append((src_path, dst_path))
                continue

        if not dry_run:
            # Atomic rename (same filesystem); also works as "move within directory"
            src_path.replace(dst_path)

        renamed.append((src_path, dst_path))

    return {"renamed": renamed, "skipped": skipped, "conflicts": conflicts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rename filename1.txt -> filename2.txt within each subdirectory of a source directory."
    )
    p.add_argument(
        "source_dir",
        help="Root directory whose subdirectories will be scanned.",
    )
    p.add_argument(
        "--src-name",
        default="filename1.txt",
        help="Source filename to look for within each subdirectory (default: filename1.txt).",
    )
    p.add_argument(
        "--dst-name",
        default="filename2.txt",
        help="Destination filename to rename to within each subdirectory (default: filename2.txt).",
    )
    p.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan immediate subdirectories (not nested).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file if it exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without making changes.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = rename_in_each_subdir(
        source_dir=args.source_dir,
        src_name=args.src_name,
        dst_name=args.dst_name,
        recursive=not args.non_recursive,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    # Summary output
    print(f"Renamed:   {len(results['renamed'])}")
    print(f"Conflicts: {len(results['conflicts'])}")
    print(f"Skipped:   {len(results['skipped'])}")

    # Detailed output
    if results["renamed"]:
        print("\nRenamed files:")
        for old, new in results["renamed"]:
            print(f"  {old} -> {new}")

    if results["conflicts"]:
        print("\nConflicts (destination exists; not moved):")
        for old, new in results["conflicts"]:
            print(f"  {old} -> {new}")

    if results["skipped"]:
        print("\nSkipped:")
        for d, reason in results["skipped"]:
            print(f"  {d}: {reason}")


if __name__ == "__main__":
    main()
#``

