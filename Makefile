PACKAGE = qnlp
PY = python
VENV = .env
BIN = $(VENV)/bin
TIN = $(TENV)/bin

all: doc .env

$(VENV): requirements.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade -r requirements.txt
	$(BIN)/pip install -e .
	touch $(VENV)

.PHONY: doc
doc: $(VENV)
	$ pdflatex -output-directory=paper paper/main.tex 


.PHONY: collect
collect: $(VENV)
	$(BIN)/python test/testTorch.py


clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete
