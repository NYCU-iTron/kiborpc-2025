on-gcp:
	python -m pip install -r $(CURDIR)/requirements.txt
	python $(CURDIR)/train.py

# connect:
# 	gcloud compute ssh --zone "europe-north1-c" "deeplearning-1-vm" --project "cogent-metric-459823-t4"

# sync-scripts:
# 	gcloud compute scp --recurse  $(CURDIR)/scripts alfonso@deeplearning-1-vm:/home/alfonso/vm/

# sync-assets:
# 	gcloud compute scp --recurse  $(CURDIR)/assets alfonso@deeplearning-1-vm:/home/alfonso/vm/

# sync-all:
# 	gcloud compute scp --recurse  $(CURDIR)/scripts alfonso@deeplearning-1-vm:/home/alfonso/vm/
# 	gcloud compute scp --recurse  $(CURDIR)/assets alfonso@deeplearning-1-vm:/home/alfonso/vm/
