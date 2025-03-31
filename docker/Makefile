# Build the image
build:
	docker build -t kiborpc-2025:latest .

# Save the image to a tar file
save:
	mkdir -p image
	docker save kiborpc-2025:latest -o image/kiborpc-2025.tar

# Load the image from a tar file
load:
	docker load -i image/kiborpc-2025.tar

# Run the container
# Note: The following command requires X11 forwarding to be set up on your host machine.
run:
# Check if kiborpc-2025 image exists
	@if [ -z "$$(docker images -q kiborpc-2025:latest)" ]; then \
		echo "Image kiborpc-2025:latest not found. Please build the image first."; \
		exit 1; \
	fi
	xhost +local:root
	-docker run -it \
		--privileged \
		--env="DISPLAY" \
		-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
		-e XDG_RUNTIME_DIR=/tmp \
		-e QT_X11_NO_MITSHM=1 \
		--net=host \
		--name kiborpc-2025 \
		--ulimit nofile=1024:524288 \
		kiborpc-2025:latest zsh
	xhost -local:root

# Start the container
start:
	-docker start kiborpc-2025
	-docker exec -it kiborpc-2025 /bin/zsh

# Stop the container
stop:
	-docker stop kiborpc-2025

# Remove the container and image
clean: stop
	-docker container rm kiborpc-2025
	-docker rmi kiborpc-2025:latest

# Attach to the running container
attach:
	-docker exec -it kiborpc-2025 /bin/zsh