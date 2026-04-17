#!/bin/bash
# Cleanup script for Math-4810_Capstone
# Purpose: discard the 846 MB push (which contains your Python .venv and would
#          be rejected by GitHub anyway due to a 111 MB .dylib), create a
#          proper .gitignore, and push ONLY your actual Model Building work.
#
# Your actual work is preserved. The .venv folder stays on your disk (you need
# it to run Python); it just won't be committed to git going forward.
#
# Run this from inside the Math-4810_Capstone folder.
# IMPORTANT: Quit Positron completely (Cmd+Q) before running this.

set -e  # Stop on first error

echo "=== 1. Removing any stale index.lock ==="
rm -f .git/index.lock

echo ""
echo "=== 2. Creating safety backup tag at current HEAD ==="
BACKUP_TAG="backup-capstone-before-cleanup-$(date +%Y%m%d-%H%M%S)"
git tag "$BACKUP_TAG"
echo "Tag created: $BACKUP_TAG"
echo "(To undo everything later: git reset --hard $BACKUP_TAG)"

echo ""
echo "=== 3. Resetting main back to origin/main ==="
echo "    (Keeps your working-tree files; just discards the 3 commits that added .venv)"
git reset --mixed origin/main

echo ""
echo "=== 4. Writing .gitignore (Python + R + Quarto + macOS) ==="
cat > .gitignore <<'EOF'
# macOS
.DS_Store

# R
.Rhistory
.Rproj.user
.RData

# Python
.venv/
venv/
env/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.ipynb_checkpoints/

# Quarto / Jupyter caches
.jupyter_cache/
.quarto/
_freeze/

# OS / editor junk
Thumbs.db
.vscode/
.idea/

# Large data files that shouldn't be in git
*.sqlite
*.sqlite3
EOF
echo "Written. Contents:"
cat .gitignore

echo ""
echo "=== 5. Staging changes ==="
echo "    (With .gitignore in place, 'git add -A' will ignore .venv automatically)"
git add .gitignore
git add -A

echo ""
echo "=== 6. Summary of what's about to be committed ==="
echo "--- File count staged: ---"
git diff --cached --name-only | wc -l
echo "--- Staged files (first 30): ---"
git diff --cached --name-only | head -30
echo "--- Any .venv files still staged? (should be ZERO) ---"
VENV_COUNT=$(git diff --cached --name-only | grep -c "\.venv/" || true)
echo "$VENV_COUNT"
if [ "$VENV_COUNT" != "0" ]; then
  echo "ERROR: .venv files are still staged. Stopping for safety."
  echo "Run: git reset --hard $BACKUP_TAG   to undo."
  exit 1
fi

echo ""
echo "=== 7. Creating a single clean commit ==="
git commit -m "Model Building work + add .gitignore"

echo ""
echo "=== 8. Verifying the new push size is reasonable ==="
RAW=$(git rev-list --objects origin/main..main | awk '{print $1}' | git cat-file --batch-check='%(objectsize)' | awk '{sum+=$1} END {print sum}')
printf "Raw push size: %.1f MB\n" "$(echo "scale=4; $RAW / 1024 / 1024" | bc)"
echo "(was 846 MB, should now be under 20 MB)"

echo ""
echo "=== 9. Checking no single file exceeds GitHub's 100 MB limit ==="
BIG=$(git rev-list --objects origin/main..main | while read sha path; do
  [ -n "$path" ] || continue
  size=$(git cat-file -s "$sha" 2>/dev/null)
  [ -z "$size" ] && continue
  [ "$size" -gt 104857600 ] && printf "%10d  %s\n" "$size" "$path"
done)
if [ -n "$BIG" ]; then
  echo "WARNING: still have files over 100 MB:"
  echo "$BIG"
  echo "GitHub will reject the push. Stop and investigate."
  exit 1
else
  echo "All files under 100 MB."
fi

echo ""
echo "=== DONE. Ready to push. ==="
echo ""
echo "Now run:   git push"
echo ""
echo "Or reopen Positron and click 'Sync Changes'."
echo ""
echo "To undo everything if needed:"
echo "    git reset --hard $BACKUP_TAG"
