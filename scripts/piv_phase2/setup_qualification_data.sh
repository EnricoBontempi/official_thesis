#!/usr/bin/env bash
# Data setup for the OPTIONAL F2.4 / F2.5 checks (authorised by roadmap sec. 10). Data live OUTSIDE the repository.
# Idempotent: skips what is already there. Python for untrusted data is run with -I (or -P); scripts live outside the data folders.
set -euo pipefail
H=$HOME/piv_qualification_data; mkdir -p "$H"/{tools,neal_unsteady_core,cardiff_d2017}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# 1. scout tools (unofficial decoder + remote-zip reader) from piv/step2-dataset-scouting @ ee9948e, copied OUT of the data folders
for f in step2_datasets_remote_zip.py step2_datasets_dynamicstudio_vectors.py step2_datasets_samples_manifest.tsv; do
  [ -f "$H/tools/$f" ] || git -C "$REPO" show ee9948e227f0d9272ae064cc8601c5e5399c28b4:mese3/02_virtual_piv/scouting/$f > "$H/tools/$f"
done
cp "$REPO"/scripts/piv_phase2/qualification_tools/*.py "$H/tools/"
# 2. Neal et al. (4TU, CC0): whole zip, 271 827 177 bytes, MD5 eb6d86228f534d6285859ccc0f16c369
N="$H/neal_unsteady_core"
if [ ! -f "$N/Unsteady Inviscid core.zip" ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$N/download_start_utc.txt"
  curl -sL -m 1800 -o "$N/Unsteady Inviscid core.zip" "https://data.4tu.nl/file/309145fd-dcd3-47c4-8b88-8349d9e70cdb/426acdf9-5762-4fad-8d99-f8604bc69c86"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$N/download_end_utc.txt"
fi
echo "eb6d86228f534d6285859ccc0f16c369  $N/Unsteady Inviscid core.zip" | md5sum -c -
[ -d "$N/extracted" ] || ( cd /tmp && python3 -I "$H/tools/extract_zip.py" "$N/Unsteady Inviscid core.zip" "$N/extracted"
  cd "$N/extracted/Unsteady Inviscid core" && for z in F001/*.zip F005/*.zip; do python3 -I "$H/tools/extract_zip.py" "$z" "${z%.zip}"; done )
# 3. OpenPIV environment (separate venv)
[ -x "$H/venv_openpiv/bin/python" ] || { python3 -m venv "$H/venv_openpiv"; "$H/venv_openpiv/bin/pip" install -q openpiv tifffile imagecodecs; }
# 4. Cardiff members are fetched by scripts/piv_phase2/f25_fetch.py (HTTP Range, resumable, CRC-32 verified)
