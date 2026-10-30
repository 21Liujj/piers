"""
Project-local utils package.

This file forces `utils` to be treated as a regular package so imports like
`import utils.sam2_bootstrap` always resolve to this repository's `utils/`
directory instead of any third-party `utils` package installed in site-packages.
"""

