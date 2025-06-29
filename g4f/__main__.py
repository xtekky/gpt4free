from __future__ import annotations

from .cli.__main__ import get_api_parser, run_api_args

parser = get_api_parser()
args = parser.parse_args()
if args.gui is None:
    args.gui = True
run_api_args(args)