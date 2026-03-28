#!/bin/bash
# Progress photo ingestion pipeline
# Usage: ./ingest.sh <date> <front.jpg> <back.jpg> <side.jpg>
# Example: ./ingest.sh 2026-03-19 /path/to/front.jpg /path/to/back.jpg /path/to/side.jpg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATE="${1:?Usage: $0 <YYYY-MM-DD> <front> <back> <side>}"
FRONT="${2:?Missing front photo}"
BACK="${3:?Missing back photo}"
SIDE="${4:?Missing side/profile photo}"

PHOTO_DIR="$SCRIPT_DIR/photos/$DATE"
ARCHIVE_DIR="$HOME/progress-photos/$DATE"
KUBECONFIG="/tmp/kubeconfig-homelab"

echo "📸 Ingesting progress photos for $DATE"

# Step 1: Archive originals
echo "  [1/5] Archiving originals..."
mkdir -p "$ARCHIVE_DIR" "$PHOTO_DIR"
cp "$FRONT" "$ARCHIVE_DIR/front.jpg"
cp "$BACK" "$ARCHIVE_DIR/back.jpg"
cp "$SIDE" "$ARCHIVE_DIR/side.jpg"
cp "$FRONT" "$PHOTO_DIR/front.jpg"
cp "$BACK" "$PHOTO_DIR/back.jpg"
cp "$SIDE" "$PHOTO_DIR/profile.jpg"

# Step 2: Add entry to data.json if not exists
echo "  [2/5] Updating data.json..."
python3 -c "
import json
from pathlib import Path

data_path = Path('$SCRIPT_DIR/data.json')
data = json.loads(data_path.read_text())

# Check if entry already exists
dates = [e['date'] for e in data['entries']]
if '$DATE' not in dates:
    data['entries'].append({
        'date': '$DATE',
        'weight': None,
        'bodyFat': None,
        'photos': {
            'front': 'photos/$DATE/front.jpg',
            'back': 'photos/$DATE/back.jpg',
            'profile': 'photos/$DATE/profile.jpg'
        },
        'notes': None
    })
    data['entries'].sort(key=lambda e: e['date'])
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print('    Entry added.')
else:
    print('    Entry already exists.')
"

# Step 3: Align photos (mediapipe + rembg)
echo "  [3/5] Aligning photos (mediapipe + rembg)..."
cd "$SCRIPT_DIR" && source venv/bin/activate && python3 align_photos.py --all 2>&1 | grep -v "^[WI]0000\|inference\|gl_context\|Tensor\|landmark\|WARNING"

# Step 4: Fetch BF from Apple Health
echo "  [4/5] Fetching body fat from Apple Health..."
if [ -f "$KUBECONFIG" ]; then
    BF=$(kubectl --kubeconfig="$KUBECONFIG" exec -n database pg-main-1 -- \
        psql -U postgres -d health -t -A -c \
        "SELECT value FROM health_metrics WHERE metric_name='body_fat_pct' AND date='$DATE' ORDER BY created_at DESC LIMIT 1;" 2>/dev/null || echo "")
    if [ -n "$BF" ]; then
        BF_ROUND=$(python3 -c "print(round(float('$BF'), 2))")
        python3 -c "
import json
from pathlib import Path
data_path = Path('$SCRIPT_DIR/data.json')
data = json.loads(data_path.read_text())
for e in data['entries']:
    if e['date'] == '$DATE':
        e['bodyFat'] = $BF_ROUND
        break
data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print(f'    BF: $BF_ROUND%')
"
    else
        echo "    No BF data yet for $DATE"
    fi
else
    echo "    Kubeconfig not found, skipping BF fetch"
fi

# Step 5: Deploy to K8s
echo "  [5/5] Deploying to K8s..."
if [ -f "$KUBECONFIG" ]; then
    POD=$(kubectl --kubeconfig="$KUBECONFIG" get pods -n progress-tracker -o jsonpath='{.items[0].metadata.name}')
    # Deploy all aligned photos (all dates, for cross-date alignment consistency)
    for d in "$SCRIPT_DIR"/photos/*/aligned; do
        PDATE=$(basename "$(dirname "$d")")
        kubectl --kubeconfig="$KUBECONFIG" exec -n progress-tracker "$POD" -- mkdir -p "/usr/share/nginx/html/photos/$PDATE/aligned"
        for f in "$d"/*.jpg; do
            [ -f "$f" ] && kubectl --kubeconfig="$KUBECONFIG" cp "$f" "$POD:/usr/share/nginx/html/photos/$PDATE/aligned/$(basename "$f")" -n progress-tracker
        done
    done
    kubectl --kubeconfig="$KUBECONFIG" cp "$SCRIPT_DIR/data.json" "$POD:/usr/share/nginx/html/data.json" -n progress-tracker
    kubectl --kubeconfig="$KUBECONFIG" cp "$SCRIPT_DIR/index.html" "$POD:/usr/share/nginx/html/index.html" -n progress-tracker
    echo "    Deployed ✅"
else
    echo "    Kubeconfig not found, skipping deploy"
fi

echo ""
echo "✅ Done! Check progress.jstrider.ovh"
