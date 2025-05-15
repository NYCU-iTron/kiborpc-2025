set-peoject-id:
	gcloud config set project dd2424-project-459820

connect:
	gcloud compute ssh --zone "us-west4-b" "deeplearning-1-vm" --project "dd2424-project-459820"

install:
	python -m pip install -r $(CURDIR)/requirements.txt

gen:
	python $(CURDIR)/gen-data.py

train:
	python $(CURDIR)/train.py

sync-script:
	gcloud compute scp --recurse  $(CURDIR)/train.py alfonso@deeplearning-1-vm:/home/alfonso/vm/

attach-tmux:
	tmux attach -t 0