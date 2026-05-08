import warnings

warnings.warn(
    "dmqclib has been renamed to aiqclib. "
    "Please update your code: pip install aiqclib. "
    "This package will not receive further updates. "
    "See https://github.com/AIQC-Hub/aiqclib for more information.",
    DeprecationWarning,
    stacklevel=2,
)

from aiqclib import *  # noqa: F401, F403, E402
from aiqclib import __version__  # noqa: E402
