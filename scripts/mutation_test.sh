#!/bin/bash
# Mutation testing of the safety-critical modules (see docs/testing.md).
#
#   scripts/mutation_test.sh            # all modules, 4 at a time
#   scripts/mutation_test.sh deid       # one module, by file name
#
# Each module runs in its own copy of the repo (mutmut 2 edits files in
# place), against only the tests that exercise it. Results land in
# $OUT/<module>.results; read a survivor with `cd $OUT/<module> && mutmut show <id>`.
# Needs: pip install "mutmut<3"  (mutmut 3 assumes src/ is not itself a package).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
MUTMUT=${MUTMUT:-$(dirname "$PY")/mutmut}
OUT=${OUT:-$(mktemp -d -t mutation-XXXX)}

declare -A TESTS=(
  [src/features/template_fields.py]="tests/test_template_fields.py tests/test_properties.py tests/test_report_release.py tests/test_web_dictation.py tests/test_field_navigation.py"
  [src/features/report_release.py]="tests/test_report_release.py tests/test_web_release.py tests/test_critical_findings.py"
  [src/features/report_checks.py]="tests/test_report_checks.py tests/test_report_release.py tests/test_web_release.py"
  [src/medical/critical_findings.py]="tests/test_critical_findings.py tests/test_report_release.py tests/test_web_release.py"
  [src/medical/deid.py]="tests/test_privacy.py tests/test_properties.py"
  [src/dictation/postprocess/voice_commands.py]="tests/test_voice_commands.py tests/test_properties.py"
  [src/medical/spelling.py]="tests/test_spelling.py tests/test_medical_dict_match.py"
)

run_one() {
  local mod=$1 tests=$2 name
  name=$(basename "$mod" .py)
  rm -rf "${OUT:?}/$name" && mkdir -p "$OUT/$name"
  (cd "$REPO" && tar --exclude=.venv --exclude=data/cache --exclude=.git -cf - .) | tar -xf - -C "$OUT/$name"
  (cd "$OUT/$name" && QT_QPA_PLATFORM=offscreen "$MUTMUT" run --paths-to-mutate "$mod" \
      --runner "$PY -m pytest -x -q -p no:cacheprovider $tests" --tests-dir tests/ > "$OUT/$name.log" 2>&1 || true
   "$MUTMUT" results > "$OUT/$name.results" 2>&1 || true)
  echo "$name: $(tail -c 300 "$OUT/$name.log" | tr '\r' '\n' | grep -v '^$' | tail -1)"
}
export -f run_one
export OUT REPO PY MUTMUT

for mod in "${!TESTS[@]}"; do
  [[ $# -eq 0 || " $* " == *" $(basename "$mod" .py) "* ]] && printf '%s\t%s\n' "$mod" "${TESTS[$mod]}"
done | xargs -P 4 -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r m t <<< "{}"; run_one "$m" "$t"'
echo "results in $OUT  (🎉 killed  🙁 survived  ⏰ timed out)"
