#!/bin/bash
# tradebot-ის მართვის პანელი. Mac-ზე: Finder-ში ორჯერ დააწკაპუნე.
# პირველ გაშვებაზე ერთხელ აყენებს ყველაფერს (.venv, ~1-2 წუთი), მერე ბრაუზერში ხსნის
# http://localhost:8501. გასაჩერებლად ეს ფანჯარა დახურე ან დააჭირე Ctrl+C.
cd "$(dirname "$0")" || exit 1

pause() { read -r -n 1 -p "დააჭირე ნებისმიერ ღილაკს ფანჯრის დასახურად…"; echo; }

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "საჭიროა Python 3.11 ან უფრო ახალი. დააყენე: brew install python@3.12"
  pause; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "პირველი გაშვება: ვაყენებ საჭირო პაკეტებს .venv-ში (ერთხელ)…"
  "$PY" -m venv .venv || { echo "venv ვერ შეიქმნა"; pause; exit 1; }
fi
if ! .venv/bin/python -c "import streamlit, ccxt, pandas" 2>/dev/null; then
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements-bot.txt || { echo "ინსტალაცია ვერ მოხერხდა (ინტერნეტი?)"; pause; exit 1; }
fi

echo "ვხსნი პანელს: http://localhost:8501"
.venv/bin/python -m tradebot dashboard "$@"
status=$?
[ $status -ne 0 ] && [ $status -ne 130 ] && pause
exit $status
