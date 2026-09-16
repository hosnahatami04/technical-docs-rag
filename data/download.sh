#!/usr/bin/env bash
#
# Download the PostgreSQL documentation corpus at a pinned version.
#
# The whole evaluation rests on this being reproducible: anyone who runs this
# script gets byte-identical source documents, so the retrieval numbers in
# results/ can be checked rather than taken on trust.
#
# Usage:  bash data/download.sh
#
set -euo pipefail

# --- The pin. Change this and every number in results/ becomes stale. -------
PG_TAG="REL_17_0"
PG_VERSION="17.0"
EXPECTED_SHA256="9a4b01944f9749e90e28b58e3c8556d900b68e3eef02ee509284d5312831787d"

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
DOCS_DIR="${RAW_DIR}/sgml"
TARBALL="${RAW_DIR}/postgres-${PG_TAG}.tar.gz"
URL="https://codeload.github.com/postgres/postgres/tar.gz/refs/tags/${PG_TAG}"

echo "PostgreSQL documentation corpus"
echo "  version : ${PG_VERSION}  (git tag ${PG_TAG})"
echo "  target  : ${DOCS_DIR}"
echo

# Already extracted? Nothing to do. Re-running must be cheap and safe.
if [[ -d "${DOCS_DIR}" ]] && [[ -n "$(ls -A "${DOCS_DIR}" 2>/dev/null)" ]]; then
    n=$(find "${DOCS_DIR}" -name '*.sgml' | wc -l)
    echo "Corpus already present (${n} .sgml files). Nothing to do."
    echo "To force a fresh download:  rm -rf ${RAW_DIR}"
    exit 0
fi

mkdir -p "${RAW_DIR}"

# --- Download ---------------------------------------------------------------
if [[ ! -f "${TARBALL}" ]]; then
    echo "Downloading ${URL} ..."
    # --fail   : a 404 must exit non-zero, not save an HTML error page
    # --location: follow redirects (codeload issues one)
    curl --fail --location --silent --show-error --output "${TARBALL}.part" "${URL}"
    mv "${TARBALL}.part" "${TARBALL}"
else
    echo "Tarball already downloaded, reusing it."
fi

# --- Verify -----------------------------------------------------------------
# A truncated or tampered download would otherwise fail much later, during
# parsing, with a confusing error. Fail loudly here instead.
echo "Verifying checksum ..."
actual=$(sha256sum "${TARBALL}" | cut -d' ' -f1)
if [[ "${EXPECTED_SHA256}" == "SKIP" ]]; then
    echo "  (checksum pinning disabled)  actual: ${actual}"
elif [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: checksum mismatch." >&2
    echo "  expected: ${EXPECTED_SHA256}" >&2
    echo "  actual  : ${actual}" >&2
    echo "The download is corrupt, or the upstream tag moved." >&2
    exit 1
else
    echo "  ok: ${actual}"
fi

# --- Extract just the documentation ----------------------------------------
# The tarball is the whole PostgreSQL source tree (~27MB); we only want the
# ~9MB of DocBook under doc/src/sgml.
echo "Extracting documentation ..."
mkdir -p "${DOCS_DIR}"
tar -xzf "${TARBALL}" \
    -C "${DOCS_DIR}" \
    --strip-components=4 \
    "postgres-${PG_TAG}/doc/src/sgml"

count=$(find "${DOCS_DIR}" -name '*.sgml' | wc -l)
echo
echo "Done. ${count} .sgml files in ${DOCS_DIR}"

# Record what we actually fetched, so the loader and the README can report the
# corpus version without anyone having to remember it.
cat > "${RAW_DIR}/CORPUS_VERSION" <<EOF
pg_version=${PG_VERSION}
git_tag=${PG_TAG}
source_url=${URL}
sha256=${actual}
sgml_files=${count}
fetched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "Wrote ${RAW_DIR}/CORPUS_VERSION"
