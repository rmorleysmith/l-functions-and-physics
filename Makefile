# make venv    create .venv with the pinned packages
# make check   run every script: each asserts the numbers it prints and redraws its figures
# make pdf     build the notes with latexmk (needs a TeX Live installation) and copy them to pdf/
PY ?= $(CURDIR)/.venv/bin/python
SCRIPTS = twelve_tones lattice_waves gauss_sums_waves light lfunction_picture golden_ratio standard_model impulses_packets light_cone zeros_statistics spheres shape_law hearing_lattices lattice_divergence
NOTES = notes/01-twelve-tones/twelve-tones \
        notes/02-characters-and-waves/characters-and-waves \
        notes/03-light/light \
        notes/04-what-an-l-function-is/what-an-l-function-is \
        notes/05-golden-ratio/golden-ratio \
        notes/06-number-theoretic-standard-model/number-theoretic-standard-model \
        notes/07-impulses-packets-spectra/impulses-packets-spectra \
        notes/08-light-cone-and-spheres/light-cone-and-spheres \
        notes/09-zeros-as-energy-levels/zeros-as-energy-levels \
        notes/10-spheres/spheres \
        notes/11-laws/laws \
        notes/12-lattice-divergence/lattice-divergence

.PHONY: venv check pdf

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

check:
	@cd scripts && for s in $(SCRIPTS); do echo "== $$s"; $(PY) $$s.py || exit 1; done

pdf:
	@mkdir -p pdf
	@for f in $(NOTES); do latexmk -pdf -cd -interaction=nonstopmode -halt-on-error $$f.tex || exit 1; cp $$f.pdf pdf/; done
