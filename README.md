# kiborpc-2025

## Usage

To enter docker build environment.

```sh
make
```

To compile apk in docker environment (The compiled apk will be generated in `app/app/build/outputs/apk/debug/`).

```sh
make build
```

To open this project in Android Studio.

```sh
make studio
```

To use doxygen to generate documentation

```sh
make doxygen
```

## Schedule

- 4/1: Simulator release
- 6/19: First round apk submit
- 7/13: Presentation

For more details, see [Progress](./docs/progress/progress.md)

## Task Distribution

- CameraHandler
  - Take pictures
  - Process the image
  - Extract and store the items information
- Navigator
  - Move to the target
  - Path planning
  - Deal with the sensor error
- Main controller
  - Determine the current state

## Useful links

- [2025競賽內容](https://2025kiborpc.ncku.edu.tw/%E7%AB%B6%E8%B3%BD%E5%85%A7%E5%AE%B9)
- [Astrobee Command API](https://nasa.github.io/astrobee/v/develop/command_dictionary.html)
- [Kibo Robot Programming Challenge official website](https://jaxa.krpc.jp/)
  - [6th Kibo-RPC Tutorial Video: 01 How to Login to My Page](https://youtu.be/PPwQDeAJsqg?si=ljjorvINLsrGOTF3)
  - [6th Kibo-RPC Tutorial Video: 02 How to Set up Android Studio](https://youtu.be/bN47LxLWkbU?si=dVKal4-G-o9Y2tIs)
  - [6th Kibo-RPC Tutorial Video: 03 How to Build APK and Simulator](https://youtu.be/LeC3sIL1sWE?si=6Vczm36ZKfC2GNsv)

## Repo of past competition

- [Kibo-RPC](https://github.com/Kobe-uni-Hyperion/Kibo-RPC)
- [kibo-2024](https://github.com/Team-Cartographer/kibo-2024)
- [kiborpc-2023](https://github.com/Team-Cartographer/kiborpc-2023)
- [3rd-Kibo-RPC_won-spaceYPublic](https://github.com/M-TRCH/3rd-Kibo-RPC_won-spaceY)
- [2ndKIbo-RPC_Indentation-Error](https://github.com/wtarit/2nd-Kibo-RPC_Indentation-Error?tab=readme-ov-file)
