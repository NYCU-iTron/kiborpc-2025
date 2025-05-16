set-peoject-id:
	gcloud config set project dd2424-project-459820

connect:
	gcloud compute ssh --zone "us-west4-b" "deeplearning-1-vm" --project "dd2424-project-459820"
	# gcloud compute ssh --zone "us-central1-c" "kibo-v100-vm" --project "dd2424-project-459820"

install:
	python -m pip install -r $(CURDIR)/requirements.txt

gen:
	python $(CURDIR)/gen-data.py
	python $(CURDIR)/train.py

sync-to-vm:
	gcloud compute scp --recurse  $(CURDIR)/train.py alfonso@deeplearning-1-vm:/home/alfonso/vm/

download-pt:
	gcloud compute scp alfonso@deeplearning-1-vm:/home/alfonso/vm/runs/detect/train/weights/best.pt $(CURDIR)/

download-all:
	gcloud compute scp alfonso@deeplearning-1-vm:/home/alfonso/vm/runs/detect/ $(CURDIR)/runs/detect --recurse

attach-tmux:
	tmux attach -t 0