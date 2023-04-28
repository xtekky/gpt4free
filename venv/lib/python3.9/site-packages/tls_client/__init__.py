#  _____  __  __         ___ _ _            _
# /__   \/ / / _\       / __\ (_) ___ _ __ | |_
#   / /\/ /  \ \ _____ / /  | | |/ _ \ '_ \| __|
#  / / / /____\ \_____/ /___| | |  __/ | | | |_
#  \/  \____/\__/     \____/|_|_|\___|_| |_|\__|

# Disclaimer:
# Big shout out to Bogdanfinn for open sourcing his tls-client in Golang.
# Also to requests, as most of the cookie handling is copied from it. :'D
# I wanted to keep the syntax as similar as possible to requests, as most people use it and are familiar with it!
# Links:
# tls-client: https://github.com/bogdanfinn/tls-client
# requests: https://github.com/psf/requests

from .sessions import Session