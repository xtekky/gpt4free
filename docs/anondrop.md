# AnonDrop file drops (`bin/anondrop`)

`bin/anondrop` uploads files to [AnonDrop](https://anondrop.io), an anonymous
file-sharing service, and prints a share link. It is a small standard-library
helper for moving logs, screenshots, or generated media off a machine without
signing up anywhere.

The API used here is documented at <https://anondrop.io/info#api>.

## Usage

```bash
# Upload one or more files (direct POST /upload)
bin/anondrop screenshot.png
bin/anondrop drop logs.zip notes.txt

# Direct file URLs instead of share pages
bin/anondrop screenshot.png --direct
bin/anondrop screenshot.png --download   # adds ?download=true

# Turn an existing share link into a direct link
bin/anondrop direct https://anondrop.io/abc123
bin/anondrop direct https://anondrop.io/abc123 --download

# Resumable chunked upload for large files (requires an account key)
bin/anondrop drop disk.img --chunked

# Let AnonDrop fetch a URL and store it (requires an account key)
bin/anondrop remote https://example.com/report.pdf

# Account operations
bin/anondrop register          # create an anonymous account, save its key
bin/anondrop list              # list files of the configured account
bin/anondrop list --direct     # direct links for every listed file
bin/anondrop delete abc123     # or pass a full share link
```

Pass `--json` to any command for machine-readable output. Upload results include
both the share page (`url`) and the direct file (`direct_url`) links. `--chunked`
shows progress on a terminal; add `--no-progress` to disable it.

`bin/anondrop FILE` is shorthand for `bin/anondrop drop FILE`.

## Direct download links

An AnonDrop share link (`https://anondrop.io/FILE_ID`) opens a page. The file
itself is served from:

```
https://anondrop.io/FILE_ID/FILENAME
```

Append `?download=true` to make browsers download it with
`Content-Disposition: attachment` instead of displaying it. The helper exposes
this in several ways:

```bash
# Upload and print the direct URL
bin/anondrop file.zip --direct

# Upload and print a forced-download URL
bin/anondrop file.zip --download

# Convert an existing share link (the filename is read from the page)
bin/anondrop direct https://anondrop.io/abc123

# Use a known filename and force a download
bin/anondrop direct abc123 --filename "report final.pdf" --download

# Direct URLs for all account files
bin/anondrop list --direct
```

JSON output always includes `direct_url` and `download_url` when the filename is
known:

```json
{
  "name": "file.zip",
  "size": 1048576,
  "url": "https://anondrop.io/abc123",
  "chunks": 1,
  "direct_url": "https://anondrop.io/abc123/file.zip",
  "download_url": "https://anondrop.io/abc123/file.zip?download=true"
}
```

Direct links also work as input to `delete`; the file id is taken from the path.

## Account keys

Direct uploads work without an account. Chunked uploads, `remote`, `list` and
`delete` need a key. The key is resolved in this order:

1. `--key KEY`
2. `$ANONDROP_KEY`
3. `$ANONDROP_KEY_FILE`, if set
4. `$XDG_CONFIG_HOME/anondrop/key` (default `~/.config/anondrop/key`)

`bin/anondrop register` creates a new anonymous account and writes the key to
the config file with mode `0600`. Use `--no-save` to print it instead and
`--key-file PATH` to store it somewhere else. Treat the key like a password: it
grants access to the account's file listing and deletion.

When an upload response issues an `X-Account-Key`, `drop --save-key` stores it.

## Environment

| Variable | Purpose |
| --- | --- |
| `ANONDROP_KEY` | Account key, overrides the key file |
| `ANONDROP_KEY_FILE` | Location of the stored key |
| `ANONDROP_URL` | Base URL, defaults to `https://anondrop.io` |

Use `--base-url` to target a mirror or onion address. The default network
timeout is 300 seconds per request; override it with `--timeout`.

## How it works

The helper mirrors the browser client:

- `drop` (default) sends a multipart `POST /upload` with the file in the `file`
  field.
- `drop --chunked` calls `GET /initiateupload`, posts 9 MiB chunks to
  `POST /uploadchunk`, then finalizes with `GET /endupload`.
- `remote` calls `GET /remoteuploadurl`.
- `list` calls `GET /files`, `delete` calls `POST /delete/FILE_ID`, and
  `register` parses the key from `GET /register`.
- `direct` reads the stored filename from the share page and builds
  `/{FILE_ID}/{FILENAME}`; `list --direct` does the same from `/files`.

AnonDrop encrypts stored files server-side and masks uploader IPs, but the share
link is public: anyone who has it can download the file. Do not drop secrets or
personal data.
