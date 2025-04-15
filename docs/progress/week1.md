# Week 1

## Meeting Record

- Task distribution
  - 典謀: Main controller, thread
  - 穎沛: Navigator
  - Rich: Train yolo model
  - 昌駿: EKF
- Using pipline or cache memory for communication between modules.
- Nameing convention: camel case for variables and functions
- To use yolo model in apk: `.pt model -> .pb model -> .tflite model, use tflite model in android app`
- To deal with the localization error, we will try EKF and simply call the api multiple times and then take the average.
- Multi-threading, the PGManual suggest don't use more than 2 threads.
  - One thread for camera handler.
  - The other thread for main controller to call navigator.

## Progress

- Complete apk compile using docker, the apk will be generated in `app/app/build/outputs/apk/debug/` folder.

## Next

- Integrate the camera handler and navigator
- Test the local simulator
- Test the apk on online simulator
