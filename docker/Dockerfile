FROM ubuntu:20.04

# Specify terminal color
ENV TERM=xterm-256color

# Install nvim
RUN apt-get update && apt-get install -y neovim

# Install zsh
RUN apt-get update && apt-get install -y zsh
RUN chsh -s $(which zsh)

# Install oh-my-zsh
RUN apt-get update && apt-get install -y curl
RUN apt-get update && apt-get install -y git
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install p10k
# Install p10k
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/g' ~/.zshrc
COPY dotfiles/.p10k.zsh /root/.p10k.zsh
COPY dotfiles/.zshrc /root/.zshrc

# copy gitstatus binary
COPY cachefile/gitstatus /root/.cache/gitstatus

# Install multiple zsh plugins
# 1. zsh-autosuggestions
# 2. zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/g' ~/.zshrc

# Install neofetch
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y neofetch

# Install qt5-default
RUN apt-get update && apt-get install -y qt5-default

# Install firefox for testing GUI
RUN apt-get update && apt-get install -y firefox

# Allow to run GUI
RUN apt-get update && apt-get install -y libpci-dev
RUN apt-get update && apt-get install -y x11-apps
RUN apt-get update && apt-get install -y qtwayland5
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yqq xserver-xorg
RUN apt-get update && apt-get install -y xwayland

# Set android environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$JAVA_HOME/bin:$ANDROID_HOME/tools:$ANDROID_HOME/tools/bin:$ANDROID_HOME/platform-tools

# Install required packages
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y unzip
RUN apt-get update && apt-get install -y gradle
RUN rm -rf /var/lib/apt/lists/*

# Install Android SDK
RUN mkdir -p $ANDROID_HOME && cd $ANDROID_HOME \
    && wget -q https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip \
    && unzip sdk-tools-linux-4333796.zip \
    && rm sdk-tools-linux-4333796.zip

# Install required Android SDK components
RUN echo "y" | $ANDROID_HOME/tools/bin/sdkmanager \
    "platforms;android-25" \
    "platforms;android-26" \
    "platforms;android-28" \
    "build-tools;25.0.3" \
    "build-tools;26.0.2" \
    "build-tools;28.0.3" \
    "platform-tools" \
    "ndk;20.0.5594570"

# Set working directory
WORKDIR /app
