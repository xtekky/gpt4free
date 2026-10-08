package main

// Version is the g4f-go release version. Bump it before building a release.
const Version = "0.1.0"

// PythonVer is the CPython version expected inside the embedded runtime archive.
// fetch-python.sh downloads exactly this version. Keep them in sync.
const PythonVer = "3.14.7"

// BundleRevision identifies the revision of the embedded Python bundle
// (g4f-go/bundle, produced by sync-bundle.sh). Bump it whenever the bundle
// changes so existing installations re-extract it on the next run.
const BundleRevision = "3"
