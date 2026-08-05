#!/usr/bin/env bash
# Fetch the 15 esp-dl ESP32-S3 model artifacts into models/zoo/ and recompute
# SHA256 for the per-model READMEs. Idempotent: re-running overwrites in place.
#
# Source: espressif/esp-dl @ 12c0616de145b704e1149c474b9a1e852e631d67 (master)
# Path: models/<name>/models/s3/*.espdl  — ALL are .espdl (no .tflite exists).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ZOO="$ROOT/models/zoo"
BASE="https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models"

# name|file1|file2|...
MODELS="
cat_detect|espdet_pico_224_224_cat.espdl|espdet_pico_416_416_cat.espdl
dog_detect|espdet_pico_224_224_dog.espdl|espdet_pico_416_416_dog.espdl
hand_detect|espdet_pico_224_224_hand.espdl
human_face_detect|espdet_pico_224_224_face.espdl|espdet_pico_416_416_face.espdl|human_face_detect_mnp_s8_v1.espdl|human_face_detect_msr_s8_v1.espdl
human_face_recognition|human_face_feat_mbf_s8_v1.espdl|human_face_feat_mfn_s8_v1.espdl
hand_gesture_recognition|mobilenetv2_0_5_128_128_gesture.espdl
pedestrian_detect|pedestrian_detect_pico_s8_v1.espdl
person_reid|person_reid_feat_osn_s8_v1.espdl
imagenet_cls|imagenet_cls_mobilenetv2_s8_v1.espdl
"

echo "$MODELS" | while IFS='|' read -r dir rest; do
  [ -z "$dir" ] && continue
  mkdir -p "$ZOO/$dir"
  for f in ${rest//|/ }; do
    url="$BASE/$dir/models/s3/$f"
    out="$ZOO/$dir/$f"
    code=$(curl -s -o "$out" -w "%{http_code}" --max-time 120 "$url")
    if [ "$code" = "200" ]; then
      sha=$(shasum -a 256 "$out" | awk '{print $1}')
      echo "OK   $dir/$f  $(stat -f%z "$out")B  $sha"
    else
      echo "FAIL $code $url"
      rm -f "$out"
    fi
  done
done
