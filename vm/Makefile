install:
	python -m pip install -r $(CURDIR)/requirements.txt
	python -m pip install -e .

gen:
	python $(CURDIR)/scripts/gen-data.py
	python $(CURDIR)/scripts/train.py

attach-tmux:
	tmux attach -t 0