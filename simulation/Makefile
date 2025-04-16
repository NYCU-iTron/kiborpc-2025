SIM_MODULE_URL = https://jaxa.krpc.jp/src/krpc6_simulator.zip
SIM_MODULE_FILE = tmp/krpc6_simulator.zip

all: download unzip docker-load edit-host run restore-host

clean:
	rm -rf tmp
	docker rmi -f krpc_astrobee_sim_local

download:
	@mkdir -p tmp
	@if [ -f $(SIM_MODULE_FILE) ]; then \
		echo -e "\e[1;36mFile $(SIM_MODULE_FILE) already exists. Skipping download.\e[0m"; \
	else \
		echo -e "\e[1;36mDownloading $(SIM_MODULE_URL) to $(SIM_MODULE_FILE) ...\e[0m"; \
		if command -v aria2c > /dev/null 2>&1; then \
			echo "Using aria2c to download $(SIM_MODULE_URL)"; \
			aria2c -x 16 -s 16 -k 1M $(SIM_MODULE_URL) -o $(SIM_MODULE_FILE); \
		elif command -v curl > /dev/null 2>&1; then \
			echo "Using curl to download $(SIM_MODULE_URL)"; \
			curl -L $(SIM_MODULE_URL) -o $(SIM_MODULE_FILE); \
		elif command -v wget > /dev/null 2>&1; then \
			echo "Using wget to download $(SIM_MODULE_URL)"; \
			wget $(SIM_MODULE_URL) -O $(SIM_MODULE_FILE); \
		else \
			echo -e "\e[1;31mError: No download tool found (aria2c, curl, wget).\e[0m"; \
			exit 1; \
		fi \
	fi

unzip:
	mkdir -p tmp

	@if [ -d tmp/local_simulator ]; then \
		echo -e "\e[1;36mDirectory tmp/local_simulator already exists. Skipping unzip.\e[0m"; \
	elif [ -f $(SIM_MODULE_FILE) ]; then \
		echo -e "\e[1;36mUnzipping $(SIM_MODULE_FILE) to tmp/local_simulator ...\e[0m"; \
		unzip $(SIM_MODULE_FILE) -d tmp; \
	else \
		echo -e "\e[1;31mError: File $(SIM_MODULE_FILE) does not exist. Please download it first.\e[0m"; \
		exit 1; \
	fi

docker-load:
	@if [ -f tmp/local_simulator/krpc_astrobee_sim_local.tar ]; then \
		echo -e "\e[1;36mLoading Docker image from tmp/local_simulator/krpc_astrobee_sim_local.tar ...\e[0m"; \
		docker load -i tmp/local_simulator/krpc_astrobee_sim_local.tar; \
	else \
		echo -e "\e[1;31mError: File tmp/local_simulator/krpc_astrobee_sim_local.tar does not exist. Please unzip it first.\e[0m"; \
		exit 1; \
	fi

edit-host:
	@echo -e "\e[1;36mEditing /etc/hosts ...\e[0m"
	chmod +x scripts/edit-host.py
	sudo scripts/edit-host.py

run:
	bash tmp/local_simulator/run.sh

restore-host:
	@echo -e "\e[1;36mRestoring /etc/hosts ...\e[0m"
	chmod +x scripts/restore-host.py
	sudo scripts/restore-host.py