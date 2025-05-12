# open zsh
all: 
	$(MAKE) -C docker all

# open android studio
studio:
	$(MAKE) -C docker all ARG="studio"

# build app
build:
	$(MAKE) -C docker all ARG="zsh -c 'cd /root/app && make build-no-daemon'"

# use doxygen to generate documentation
doxygen:
	@if [ -z $$(which doxygen) ]; then \
		echo -e "\033[31mDoxygen not installed. Please install doxygen first.\033[0m"; \
		exit 1; \
	fi
	doxygen Doxyfile
	xdg-open docs/doxygen/html/index.html

# train yolo model
train:
	source venv/bin/activate && $(MAKE) -C pyproto/yolo-script all