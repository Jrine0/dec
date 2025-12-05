
import os

file_path = "omr_engine.py"

with open(file_path, "r") as f:
    lines = f.readlines()

# Find the start of the corrupted block
# It starts after "print(f"DEBUG: Column {col_idx} has {len(rows)} rows (Global Grid)", flush=True)"
start_idx = -1
for i, line in enumerate(lines):
    if "DEBUG: Column {col_idx} has {len(rows)} rows (Global Grid)" in line:
        start_idx = i
        break

if start_idx == -1:
    print("Could not find start index")
    exit(1)

# Find the end of the corrupted block
# It ends before "# Adaptive Peak Mapping"
end_idx = -1
for i in range(start_idx, len(lines)):
    if "# Adaptive Peak Mapping" in line: # Wait, line variable is stale
        pass

for i in range(start_idx, len(lines)):
    if "# Adaptive Peak Mapping" in lines[i]:
        end_idx = i
        break

if end_idx == -1:
    print("Could not find end index")
    # Fallback: look for "if not peaks:"
    for i in range(start_idx, len(lines)):
        if "if not peaks:" in lines[i]:
            end_idx = i - 4 # Approximate
            break

print(f"Replacing lines {start_idx+1} to {end_idx}")

# The correct code block
correct_code = [
    "\n",
    "            # Calculate column bounds using Histogram Peaks (Robust to heavy noise)\n",
    "            all_cx = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]//2 for c in col]\n",
    "            if not all_cx: continue\n",
    "            \n",
    "            # Create histogram\n",
    "            hist_range = (0, img.shape[1])\n",
    "            hist_bins = int(img.shape[1] / 5)\n",
    "            hist, bin_edges = np.histogram(all_cx, bins=hist_bins, range=hist_range)\n",
    "            \n",
    "            # Find all peaks\n",
    "            peaks = []\n",
    "            peak_heights = []\n",
    "            threshold = max(hist) * 0.2 # Lower threshold to catch all\n",
    "            for i in range(1, len(hist)-1):\n",
    "                if hist[i] > threshold and hist[i] > hist[i-1] and hist[i] > hist[i+1]:\n",
    "                    peaks.append((bin_edges[i] + bin_edges[i+1]) / 2)\n",
    "                    peak_heights.append(hist[i])\n",
    "            \n",
    "            print(f\"DEBUG: Col {col_idx} ALL Peaks: {list(zip(peaks, peak_heights))}\", flush=True)\n",
    "            \n"
]

new_lines = lines[:start_idx+1] + correct_code + lines[end_idx:]

with open(file_path, "w") as f:
    f.writelines(new_lines)

print("Fixed.")
