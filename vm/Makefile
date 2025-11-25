install:
	uv sync

run:
	uv run scripts/generate_dataset.py
	uv run scripts/train_model.py

attach-tmux:
	tmux attach -t 0

rsync:
	rsync -avz --progress --exclude-from='rsync_exclude.txt' \
		-e "ssh -i ~/.ssh/kthcloud -p 20055" \
		./ root@vm.cloud.cbh.kth.se:~/vm/

pull-runs:
	rsync -avz --progress \
		-e "ssh -i ~/.ssh/kthcloud -p 20055" \
		root@vm.cloud.cbh.kth.se:~/vm/runs .
