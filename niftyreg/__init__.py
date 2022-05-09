"""Thin wrapper around niftyreg binaries"""
__author__ = "Casper da Costa-Luis <https://github.com/casperdcl>"
__date__ = "2022"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError: # pragma: nocover
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
__all__ = ['bin_path', 'main']

from pathlib import Path

bin_path = Path(__file__).resolve().parent / "bin"


def main(args=None):
    if args is None:
        import sys
        args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(f"Options: {' '.join(i.name[4:] for i in bin_path.glob('reg_*'))}")
    else:
        from subprocess import run
        run([str(bin_path / ("reg_" + args[0]))] + args[1:])
