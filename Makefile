# make venv    create .venv with the pinned packages
# make check   run every script: each asserts the numbers it prints and redraws its figures
# make pdf     build the notes with latexmk (needs a TeX Live installation) and copy them to pdf/
PY ?= $(CURDIR)/.venv/bin/python
SCRIPTS = twelve_tones lattice_waves gauss_sums_waves light
NOTES = notes/01-twelve-tones/twelve-tones \
        notes/02-characters-and-waves/characters-and-waves \
        notes/03-light/light

.PHONY: venv check pdf

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

check:
	@cd scripts && for s in $(SCRIPTS); do echo "== $$s"; $(PY) $$s.py || exit 1; done

pdf:
	@mkdir -p pdf
	@for f in $(NOTES); do latexmk -pdf -cd -interaction=nonstopmode -halt-on-error $$f.tex || exit 1; cp $$f.pdf pdf/; done
