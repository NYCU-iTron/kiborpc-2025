ARG ?= zsh

all:
### Build
	make ensure-build
### Run
	make container-rm
	make run ARG=$(ARG)

###############################
### Docker Builtin Commands ###
###############################

# Build the image
build:
	docker build -t kiborpc-2025:latest .

# Save the image to a tar file
save:
	mkdir -p tmp/image
	docker save kiborpc-2025:latest -o tmp/image/kiborpc-2025.tar
	@echo "Image saved to tmp/image/kiborpc-2025.tar"

# Load the image from a tar file
load:
	docker load -i tmp/image/kiborpc-2025.tar

# Run the container
# Note: The following command requires X11 forwarding to be set up on your host machine.
run:
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
		--mount type=bind,source=$(CURDIR)/../app,target=/app \
		kiborpc-2025:latest $(ARG)
	xhost -local:root

# Start the container
start:
	-docker start kiborpc-2025
	-docker exec -it kiborpc-2025 /bin/zsh

# Stop the container
stop:
	-docker stop kiborpc-2025

container-rm:
	-docker container rm kiborpc-2025

rmi:
	-docker rmi kiborpc-2025:latest

# Attach to the running container
attach:
	-docker exec -it kiborpc-2025 /bin/zsh

#####################
### Tool Commands ###
#####################

# Remove the container and image
clean: stop container-rm rmi
	echo "Cleaned up the container and image."

### Build
# Build the image if network is present
#     If the cached image is different from the current image, replace the cached image
#     If the cached image is the same as the current image, do nothing
# If not, find the cached image
# If cached image is not present, check if the image is already present
# If all fail, exit with error code
ensure-build:
	@if ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1; then \
		echo -e "\e[1;36mNetwork is up, building the image\e[0m"; \
		make build; \
		if [ -f "tmp/dockerfile-hash.txt" ] && sha256sum -c tmp/dockerfile-hash.txt > /dev/null 2>&1; then \
			echo -e "\e[1;36mCached image is up to date\e[0m"; \
		else \
			echo -e "\e[1;36mCached image is different, replacing the cached image\e[0m"; \
			make save; \
			sha256sum Dockerfile > tmp/dockerfile-hash.txt; \
		fi; \
	else \
		echo -e "\e[1;36mNetwork is down, finding the cached image\e[0m"; \
		if [ -f "tmp/image/kiborpc-2025.tar" ]; then \
			echo -e "\e[1;36mCached image found, loading the image\e[0m"; \
			make load; \
		else \
			echo -e "\e[1;36mCached image not found, checking if the image is already present\e[0m"; \
			if [ -z "$$(docker images -q kiborpc-2025:latest)" ]; then \
				echo -e "\e[31mImage not found, please build the image when connected to a network\e[0m"; \
				exit 1; \
			else \
				echo -e "\e[1;36mImage found in docker.\e[0m"; \
			fi; \
		fi; \
	fi