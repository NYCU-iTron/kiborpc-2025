# open zsh
all: 
	$(MAKE) -C docker all

# open android studio
studio:
	$(MAKE) -C docker all ARG="studio"

# build app
build:
	$(MAKE) -C docker all ARG="zsh -c 'cd /root/app && make build-no-daemon'"