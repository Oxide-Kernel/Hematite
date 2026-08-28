#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_all.sh — unified QEMU test runner for Hematite.
#
# Builds BOTH QEMU-runnable benchmark suites, merges each at a configurable
# flash size, runs each under the Espressif QEMU fork with `-icount 3` (or
# free-running with `--no-icount`), and validates the serial logs against the
# documented golden checksums.
#
#   Suite 1  freestanding C baseline   benchmarks/qemu-baseline (make)
#   Suite 2  Rust firmware             hematite-benchmarks --features qemu
#                                     (--models adds qemu,model-validation)
#
# Exit status: 0 only when every expected row PASSes.
#
# Env overrides: QEMU_BIN, ESPFLASH, XT_GCC, ESP_TOOLCHAIN_DIR.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOGS_DIR="$SCRIPT_DIR/logs"
IMAGES_DIR="$SCRIPT_DIR/images"
BASELINE_DIR="$REPO_ROOT/benchmarks/qemu-baseline"
RUST_ELF="$REPO_ROOT/target/xtensa-esp32s3-none-elf/release/hematite-benchmarks"

mkdir -p "$LOGS_DIR" "$IMAGES_DIR"

# ── defaults (env-overridable tool paths) ─────────────────────────────────
QEMU_BIN="${QEMU_BIN:-$HOME/.esp-qemu/qemu/bin/qemu-system-xtensa}"
ESPFLASH="${ESPFLASH:-$HOME/.cargo/bin/espflash}"
ESP_TOOLCHAIN_DIR="${ESP_TOOLCHAIN_DIR:-$HOME/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin}"
XT_GCC="${XT_GCC:-$ESP_TOOLCHAIN_DIR/xtensa-esp32s3-elf-gcc}"

# ── flags ─────────────────────────────────────────────────────────────────
FLASH_SIZE="4mb"
DO_MODELS=0
DO_PSRAM=0
SKIP_C=0
SKIP_RUST=0
DUR_C=30          # seconds (C baseline)
DUR_RUST=120      # seconds (Rust firmware; documented ~90s lands at the edge —
                  # the completion marker prints ~1 line after the last row)
DUR_MV=180        # seconds (model-validation suite; documented ~150s + headroom)
TOLERATE="auto"   # auto: ON for model-validation, OFF for the other suites
FAST=0
NO_REBUILD=0
NO_ICOUNT=0       # 1: drop `-icount 3` (free-running; cycles become informational)

QEMU_PID=""

usage() {
    cat <<'EOF'
run_all.sh — unified QEMU test runner for Hematite

Builds the freestanding C baseline (benchmarks/qemu-baseline) and the Rust
benchmark firmware (hematite-benchmarks, `qemu` feature), merges each at a
configurable flash size, runs each under the Espressif QEMU fork
(-icount 3 by default, serial to logs/), and validates the output against the
documented golden checksums (see benchmarks/QEMU_VALIDATION.md). Prints a
unified PASS/FAIL table; exits non-zero on unexpected failure.

Usage: run_all.sh [flags]

Flags:
  --flash-size SIZE       2mb | 4mb | 8mb | 16mb  (default 4mb; QEMU only
                          accepts these drive-image sizes)
  --models                additionally build+run the model-validation suite
                          (features qemu,model-validation; ~150s default run)
  --psram                 add `-m 8M` to QEMU (attaches emulated PSRAM)
  --no-icount             drop `-icount 3` (free-running; ~30x+ faster, full
                          suite runs in seconds). Cycles no longer match the
                          documented goldens (they only match at icount 3)
                          and are reported as drift -- checksums still
                          validate bit-exact. Default: icount 3.
  --skip-c                skip the C baseline suite
  --skip-rust             skip the plain Rust suite (model-validation suite,
                          when enabled, still runs)
  --durations C R MV      run durations in seconds (default "30 120 180";
                          the documented ~90s Rust / ~150s MV values land at
                          the edge and can miss the completion marker)
  --tolerate-divergences  allow documented-divergent rows (person_detect FAIL;
                          default ON for model-validation, OFF otherwise)
  --no-tolerate-divergences
                          force strict validation on every row
  --fast                  skip rebuilds when artifacts are up to date
                          (C: `make -q baseline.elf`; Rust: stamp + source
                          mtime freshness)
  --no-rebuild            never rebuild; fail if required artifacts are missing
  --help                  show this text

Env overrides (defaults shown):
  QEMU_BIN           ~/.esp-qemu/qemu/bin/qemu-system-xtensa
  ESPFLASH           ~/.cargo/bin/espflash
  ESP_TOOLCHAIN_DIR  ~/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin
  XT_GCC             $ESP_TOOLCHAIN_DIR/xtensa-esp32s3-elf-gcc

Validation rule: a row PASSes iff every expected checksum/fnv value appears
in its serial log AND the suite's completion marker appears. Cycle minima are
informational only (5% tolerance) and never gate the verdict. With
`--no-icount` the cycle minima no longer match the documented goldens (those
are locked to `-icount 3`) and always print as drift.

Examples:
  run_all.sh                          # C + Rust at 4mb
  run_all.sh --flash-size 16mb --psram
  run_all.sh --no-icount              # free-running: full suite in seconds
  run_all.sh --models --flash-size 8mb --durations 30 120 180
  run_all.sh --skip-rust --fast       # C only, reuse fresh artifacts
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

cleanup() {
    [ -n "$QEMU_PID" ] && kill -9 "$QEMU_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ── arg parsing ───────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --flash-size)
            FLASH_SIZE="$2"; shift 2 ;;
        --models)
            DO_MODELS=1; shift ;;
        --psram)
            DO_PSRAM=1; shift ;;
        --no-icount)
            NO_ICOUNT=1; shift ;;
        --skip-c)
            SKIP_C=1; shift ;;
        --skip-rust)
            SKIP_RUST=1; shift ;;
        --durations)
            [ $# -ge 4 ] || die "--durations needs 3 values (C R MV), e.g. --durations 30 120 180"
            DUR_C="$2"; DUR_RUST="$3"; DUR_MV="$4"; shift 4 ;;
        --tolerate-divergences)
            TOLERATE="on"; shift ;;
        --no-tolerate-divergences)
            TOLERATE="off"; shift ;;
        --fast)
            FAST=1; shift ;;
        --no-rebuild)
            NO_REBUILD=1; shift ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            die "unknown flag: $1 (try --help)" ;;
    esac
done

# ── validation of inputs ──────────────────────────────────────────────────
case "$FLASH_SIZE" in
    2mb|4mb|8mb|16mb) ;;
    *) die "--flash-size must be 2mb|4mb|8mb|16mb (got '$FLASH_SIZE')" ;;
esac

for v in "$DUR_C" "$DUR_RUST" "$DUR_MV"; do
    case "$v" in
        ''|*[!0-9]*) die "--durations values must be integers (got '$v')" ;;
    esac
done

[ "$SKIP_C" = 1 ] && [ "$SKIP_RUST" = 1 ] && [ "$DO_MODELS" = 0 ] \
    && die "nothing to run (--skip-c + --skip-rust, no --models)"

# ── tool existence checks ─────────────────────────────────────────────────
[ -x "$QEMU_BIN" ]  || die "QEMU not found: $QEMU_BIN (set QEMU_BIN)"
[ -x "$ESPFLASH" ]  || die "espflash not found: $ESPFLASH (set ESPFLASH)"
[ -x "$XT_GCC" ]    || die "xtensa gcc not found: $XT_GCC (set XT_GCC / ESP_TOOLCHAIN_DIR)"
[ -f "$HOME/export-esp.sh" ] || echo "WARN: ~/export-esp.sh missing (Rust build may fail)" >&2

# ── helpers ───────────────────────────────────────────────────────────────

# run_qemu <image> <log> <seconds>: launch QEMU in background, wait, kill.
run_qemu() {
    local img="$1" log="$2" secs="$3"
    local qemu=("$QEMU_BIN" -nographic -machine esp32s3)
    [ "$DO_PSRAM" = 1 ] && qemu+=(-m 8M)
    qemu+=(-drive "file=$img,if=mtd,format=raw" -monitor none -serial "file:$log")
    # `-icount 3` forces deterministic-CCOUNT mode, the dominant QEMU
    # slowdown (~30x+ wall vs free-running); only it reproduces the
    # documented golden cycle minima. `--no-icount` drops it: checksums stay
    # bit-exact but cycles drift (informational) -- see QEMU_VALIDATION.md §8.
    [ "$NO_ICOUNT" = 1 ] || qemu+=(-icount 3)

    echo "  qemu: $("$QEMU_BIN" --version 2>/dev/null | head -1)"
    echo "  opts: ${qemu[*]}"
    "${qemu[@]}" &
    QEMU_PID=$!
    echo "  QEMU PID $QEMU_PID; sleeping ${secs}s..."
    sleep "$secs"
    kill -INT "$QEMU_PID" 2>/dev/null
    local i=0
    while kill -0 "$QEMU_PID" 2>/dev/null && [ $i -lt 10 ]; do
        sleep 1; i=$((i + 1))
    done
    kill -9 "$QEMU_PID" 2>/dev/null
    wait "$QEMU_PID" 2>/dev/null
    QEMU_PID=""
    [ -s "$log" ] || die "serial log empty/missing: $log — QEMU did not boot?"
}

# absdiff <a> <b> -> |a-b|
absdiff() {
    if [ "$1" -ge "$2" ]; then echo "$(( $1 - $2 ))"; else echo "$(( $2 - $1 ))"; fi
}

# within_tol <got> <want> <pct> -> 0 if within pct tolerance, else 1
within_tol() {
    local got="$1" want="$2" pct="$3" tol
    tol=$(( want * pct / 100 ))
    [ "$(absdiff "$got" "$want")" -le "$tol" ]
}

# Cycle-minima check helper: compare got vs want (informational).
# Returns via echo: "ok" | "drift(kernel: got vs want)"
cyc_verdict() {
    local name="$1" got="$2" want="$3"
    if [ -n "$got" ] && within_tol "$got" "$want" 5; then
        echo "ok"
    else
        echo "drift($name: ${got:-missing} vs $want)"
    fi
}

# ── C baseline: build ─────────────────────────────────────────────────────
build_c() {
    echo "== C baseline build (benchmarks/qemu-baseline) =="
    if [ "$NO_REBUILD" = 1 ]; then
        [ -f "$BASELINE_DIR/baseline.elf" ] || die "--no-rebuild but baseline.elf missing"
        echo "  --no-rebuild: reusing baseline.elf"
    elif [ "$FAST" = 1 ] && make -C "$BASELINE_DIR" -q baseline.elf XT_GCC="$XT_GCC" 2>/dev/null; then
        echo "  --fast: baseline.elf up to date, skipping make"
    else
        make -C "$BASELINE_DIR" baseline.elf XT_GCC="$XT_GCC" || die "C baseline build failed"
    fi
}

# ── Rust firmware: build ──────────────────────────────────────────────────
# rust_stamp tracks which feature set the ELF was built with, so --fast
# rebuilds when the requested features differ.
rust_stamp() { echo "$IMAGES_DIR/.rust_features"; }

rust_needs_build() {
    local elf="$1" want="$2" stamp
    stamp=$(rust_stamp)
    [ -f "$elf" ] || { echo "  ELF missing"; return 0; }
    [ "$(cat "$stamp" 2>/dev/null)" = "$want" ] || { echo "  feature set changed (stamp: $(cat "$stamp" 2>/dev/null) vs $want)"; return 0; }
    # any .rs / Cargo.toml / Cargo.lock / build.rs newer than the ELF?
    if find "$REPO_ROOT" \
        \( -path "$REPO_ROOT/target" -o -path "$REPO_ROOT/.git" -o -path "$REPO_ROOT/local-notes" \) -prune -o \
        -type f \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' -o -name 'build.rs' \) \
        -newer "$elf" -print -quit 2>/dev/null | grep -q .; then
        echo "  source newer than ELF"
        return 0
    fi
    return 1
}

build_rust() {
    local features="$1" elf="$2"
    echo "== Rust firmware build (features: $features) =="
    if [ "$NO_REBUILD" = 1 ]; then
        [ -f "$elf" ] || die "--no-rebuild but $elf missing"
        # cargo xtensa-build writes to the SAME ELF path for every feature
        # set — the stamp records which build the ELF currently is, and a
        # mismatched --no-rebuild would silently run the wrong firmware.
        if [ "$(cat "$(rust_stamp)" 2>/dev/null)" != "$features" ]; then
            die "--no-rebuild but ELF was built with '$(cat "$(rust_stamp)" 2>/dev/null)' != requested '$features' — drop --no-rebuild (or use --fast)"
        fi
        echo "  --no-rebuild: reusing $elf"
        return 0
    fi
    if [ "$FAST" = 1 ] && ! rust_needs_build "$elf" "$features"; then
        echo "  --fast: $elf up to date, skipping cargo"
        return 0
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    . "$HOME/export-esp.sh" >/dev/null 2>&1
    (cd "$REPO_ROOT" && cargo xtensa-build --release -p hematite-benchmarks --features "$features") \
        || die "Rust firmware build failed (features: $features)"
    echo "$features" > "$(rust_stamp)"
}

# ── merge + run + validate per suite ──────────────────────────────────────
# Each validate_* function prints one table row and returns 0/1 (PASS/FAIL).

validate_c() {
    local log="$1" fail=0 checks=0 total=6
    local row="| c-baseline | $FLASH_SIZE |"

    local qopts="free-running"
    [ "$NO_ICOUNT" = 1 ] || qopts="-icount 3"
    [ "$DO_PSRAM" = 1 ] && qopts="$qopts -m 8M"
    row="$row $qopts |"

    # boot marker (proves .data copy) + 4 kernel checksums + completion marker
    grep -qF 'boot_marker (.data copy check)=0x0badc0de' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING boot_marker=0x0badc0de" >&2; fail=1; }
    grep -qF 'out_checksum(fnv1a)=0xcc7be479' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING conv_s8 checksum 0xcc7be479" >&2; fail=1; }
    grep -qF 'out_checksum(fnv1a)=0x49f763d2' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING depthwise checksum 0x49f763d2" >&2; fail=1; }
    grep -qF 'out_checksum(fnv1a)=0xc87e9c19' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING fc checksum 0xc87e9c19" >&2; fail=1; }
    grep -qF 'out_checksum(fnv1a)=0x272a7025' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING conv1x1 checksum 0x272a7025" >&2; fail=1; }
    grep -qF '=== benchmark complete - QEMU halt ===' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING completion marker (benchmark complete)" >&2; fail=1; }

    row="$row $checks/$total |"

    # informational cycle minima (documented goldens, 5% tolerance)
    local mins got c
    mins=$(grep 'min    cycles=0x' "$log" | head -4)
    c=""
    got=$(echo "$mins" | sed -n '1s/.*min    cycles=0x\([0-9a-fA-F]*\).*/\1/p');   [ -n "$got" ] && got=$((16#$got));   c="$c $(cyc_verdict conv_s8 "$got" 737287)"
    got=$(echo "$mins" | sed -n '2s/.*min    cycles=0x\([0-9a-fA-F]*\).*/\1/p');   [ -n "$got" ] && got=$((16#$got));   c="$c $(cyc_verdict depthwise "$got" 572272)"
    got=$(echo "$mins" | sed -n '3s/.*min    cycles=0x\([0-9a-fA-F]*\).*/\1/p');   [ -n "$got" ] && got=$((16#$got));   c="$c $(cyc_verdict fc "$got" 2425)"
    got=$(echo "$mins" | sed -n '4s/.*min    cycles=0x\([0-9a-fA-F]*\).*/\1/p');   [ -n "$got" ] && got=$((16#$got));   c="$c $(cyc_verdict conv1x1 "$got" 14793)"

    if [ "$fail" = 0 ]; then
        echo "$row$c | PASS |"
        return 0
    else
        echo "$row$c | FAIL |"
        return 1
    fi
}

validate_rust() {
    local log="$1" suite_label="$2" mv_suite="${3:-0}" fail=0 checks=0 total=6
    local row="| $suite_label | $FLASH_SIZE |"

    local qopts="free-running"
    [ "$NO_ICOUNT" = 1 ] || qopts="-icount 3"
    [ "$DO_PSRAM" = 1 ] && qopts="$qopts -m 8M"
    row="$row $qopts |"

    # 4 kernel fnv + Model A + Model B
    grep -qF '0xa6d4f279' "$log" && checks=$((checks + 1)) || { echo "  MISSING conv_s8 fnv 0xa6d4f279" >&2; fail=1; }
    grep -qF '0x56e836d2' "$log" && checks=$((checks + 1)) || { echo "  MISSING depthwise fnv 0x56e836d2" >&2; fail=1; }
    grep -qF '0x7d803f19' "$log" && checks=$((checks + 1)) || { echo "  MISSING fc_s8 fnv 0x7d803f19" >&2; fail=1; }
    grep -qF '0x0bea8225' "$log" && checks=$((checks + 1)) || { echo "  MISSING conv1x1 fnv 0x0bea8225" >&2; fail=1; }
    grep -qE 'cnn_model 4-layer.*0x75eb32f5' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING Model A out_fnv 0x75eb32f5 (cnn_model)" >&2; fail=1; }
    grep -qE 'mv2mini 7-layer.*0x7f23eb05' "$log" \
        && checks=$((checks + 1)) || { echo "  MISSING Model B out_fnv 0x7f23eb05 (mv2mini)" >&2; fail=1; }

    # Completion marker: the model-validation suite prints
    # "=== MODEL VALIDATION DONE ===" (reachable). The PLAIN Rust suite has NO
    # reachable end marker on current firmware: after the four ember rows the
    # kernel loop continues into the conv3x3_s8 32x32 SIMD row, which grinds
    # for HOURS in scalar fallback under QEMU at `-icount 3` (documented in
    # benchmarks/QEMU_VALIDATION.md §4) — the "benchmarks complete; reference
    # bars:" line at firmware.rs:1556 is never reached. Free-running
    # (--no-icount) the row completes in seconds and the marker IS reached.
    # The six fnv values above are the completion proof in both modes.
    if [ "$mv_suite" = 1 ]; then
        total=7
        grep -qF '=== MODEL VALIDATION DONE ===' "$log" \
            && checks=$((checks + 1)) || { echo "  MISSING MODEL VALIDATION DONE marker" >&2; fail=1; }
    fi

    row="$row $checks/$total |"

    # informational cycle minima (documented release goldens, 5% tolerance)
    local mins got c
    mins=$(grep -E '^\| (conv_s8 8x8,64x3x3x3|depthwise_conv_s8 18x18,1x3x3x16|fc_s8 271row,3out|conv1x1_s8 64x1x1x64)' "$log")
    c=""
    got=$(echo "$mins" | grep '^| conv_s8 '    | head -1 | sed 's/.*| SRAM | \([0-9]*\)\/.*/\1/'); c="$c $(cyc_verdict conv_s8 "$got" 1045880)"
    got=$(echo "$mins" | grep '^| depthwise_' | head -1 | sed 's/.*| SRAM | \([0-9]*\)\/.*/\1/'); c="$c $(cyc_verdict depthwise "$got" 744236)"
    got=$(echo "$mins" | grep '^| fc_s8 '     | head -1 | sed 's/.*| SRAM | \([0-9]*\)\/.*/\1/'); c="$c $(cyc_verdict fc "$got" 3498)"
    got=$(echo "$mins" | grep '^| conv1x1_s8' | head -1 | sed 's/.*| SRAM | \([0-9]*\)\/.*/\1/'); c="$c $(cyc_verdict conv1x1 "$got" 25981)"

    # documented divergence: person_detect FAIL (matches hardware). Tolerated
    # by default for the model-validation suite.
    if [ "$mv_suite" = 1 ]; then
        if grep -qE 'model person_detect_int8: FAIL' "$log"; then
            echo "  NOTE: person_detect_int8 FAIL (documented divergence, matches hardware)" >&2;
            if [ "$TOLERATE" = "off" ]; then
                echo "  --no-tolerate-divergences: person_detect FAIL is a hard failure" >&2;
                fail=1
            fi
        fi
    fi

    if [ "$fail" = 0 ]; then
        echo "$row$c | PASS |"
        return 0
    else
        echo "$row$c | FAIL |"
        return 1
    fi
}

# save_image <elf> <out> <suite_label>: espflash merge with the configured
# flash size; adds a helpful hint when the app image exceeds the builtin
# factory partition for the requested size.
save_image() {
    local elf="$1" out="$2" label="$3"
    if ! "$ESPFLASH" save-image --chip esp32s3 --merge --flash-size "$FLASH_SIZE" \
            "$elf" "$out" >/dev/null; then
        echo "ERROR: save-image failed for $label at ${FLASH_SIZE}." >&2
        echo "  The app image must fit the builtin factory partition" >&2
        echo "  (${FLASH_SIZE}: 4,128,768 B / 8mb: 8,323,072 B / 16mb: 16,384,000 B)." >&2
        echo "  If the image is too big, use a larger --flash-size." >&2
        exit 1
    fi
}

# ── main ──────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
echo "=== Hematite QEMU runner — flash $FLASH_SIZE, $(date) ==="
echo "logs: $LOGS_DIR/$TS"
echo
echo "| suite | flash | QEMU opts | checksums | cycles | verdict |"
echo "|-------|-------|-----------|-----------|--------|---------|"

PASSED=0
FAILED=0

# ── suite 1: C baseline ───────────────────────────────────────────────────
if [ "$SKIP_C" = 0 ]; then
    build_c
    C_IMG="$IMAGES_DIR/baseline_${FLASH_SIZE}.bin"
    echo "== merging C baseline at $FLASH_SIZE =="
    save_image "$BASELINE_DIR/baseline.elf" "$C_IMG" "C baseline"
    C_LOG="$LOGS_DIR/${TS}_c_${FLASH_SIZE}.log"
    echo "== running C baseline (${DUR_C}s) =="
    run_qemu "$C_IMG" "$C_LOG" "$DUR_C"
    echo "== validating C log =="
    if validate_c "$C_LOG"; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
fi

# ── suite 2: Rust firmware (qemu feature) ─────────────────────────────────
if [ "$SKIP_RUST" = 0 ]; then
    build_rust "qemu" "$RUST_ELF"
    R_IMG="$IMAGES_DIR/rust_${FLASH_SIZE}.bin"
    echo "== merging Rust firmware at $FLASH_SIZE =="
    save_image "$RUST_ELF" "$R_IMG" "Rust firmware"
    R_LOG="$LOGS_DIR/${TS}_rust_${FLASH_SIZE}.log"
    echo "== running Rust firmware (${DUR_RUST}s) =="
    run_qemu "$R_IMG" "$R_LOG" "$DUR_RUST"
    echo "== validating Rust log =="
    if validate_rust "$R_LOG" "rust(qemu)"; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
fi

# ── suite 3: model-validation (qemu,model-validation) ─────────────────────
if [ "$DO_MODELS" = 1 ]; then
    if [ "$TOLERATE" = "auto" ]; then
        echo "NOTE: --tolerate-divergences auto-ON for model-validation suite"
        TOLERATE="on"
    fi
    build_rust "qemu,model-validation" "$RUST_ELF"
    MV_IMG="$IMAGES_DIR/rust_mv_${FLASH_SIZE}.bin"
    echo "== merging model-validation firmware at $FLASH_SIZE =="
    save_image "$RUST_ELF" "$MV_IMG" "model-validation firmware"
    MV_LOG="$LOGS_DIR/${TS}_rust_mv_${FLASH_SIZE}.log"
    echo "== running model-validation firmware (${DUR_MV}s) =="
    run_qemu "$MV_IMG" "$MV_LOG" "$DUR_MV"
    echo "== validating model-validation log =="
    if validate_rust "$MV_LOG" "rust(qemu,model-validation)" 1; then
        PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
fi

# ── summary ───────────────────────────────────────────────────────────────
echo
if [ "$FAILED" = 0 ] && [ "$PASSED" -gt 0 ]; then
    echo "ALL PASS ($PASSED suite(s))"
    exit 0
else
    echo "FAILURES: $FAILED suite(s) failed, $PASSED passed"
    exit 1
fi
