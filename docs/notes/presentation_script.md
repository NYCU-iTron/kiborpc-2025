# Presentation Script

## Introduction (25 sec)

Hello everyone, thank you for being here today.
I’m 林穎沛 from the Itron Robotics Team at National Chiao Tung University.

Today, I’ll walk you through how we built our system for the competition, including code structure, image pipeline, major challenges we faced and finally, the results we achieved in simulation.

## Code structure (50 sec)

Our program is modular, with each component handling a specific task.

ItemManager is responsible for storing detected item data.

VisionHandler manages image processing and includes
ArTagDetector, for reading AR tag data from images,
CameraHandler, to interface with the camera and manage calibration,
and ItemDetector, which applies our YOLO model.

Navigator is in charge of planning and executing movements.

This structure makes it easier to debug, and also allows for flexibility in testing different models or detection strategies.

## Image pipeline (60 sec)

Here’s how our image pipeline works.

First, we use camera handler to get the raw image, then distort the image based on the camera calibration parameters.

After, the undistorted image is passed to the artag detector to extract the AR tag information and use that to crop the image to the area of interest.

Next, the cropped image is passed to the item detector, which uses a yolo model to detect items. Preprocessing steps such as resizing and normalization make sure the image is in the correct format to the model.
After the reference, the model returns a list of detected items with their bounding boxes and confidence scores.

Then, in postprocess, we filter out low confidence detections to ensure only reliable items are considered by setting a threshold.

Finally, we apply weighted box fusion instead of non-maximum suppression to merge overlapping bounding boxes for more accurate results.

## Challenge and Solution (60 sec)

Of course, the process wasn't without obstacles.

First, Programming Mannual has no detail about KiboRpcService class, we couldn’t import service api properly at first. We ended up decompiling the sample APK using Android Studio, which helped us understand how the class worked and integrate it into our project.

Second, the online simulator sometimes fail without any log, or any error message, which make us extremely hard to find the bug. We had to test repeatedly, narrowing down the error line by line through try and error.

Third, training yolo models on local laptop is extremely slow. We reached out to professors and some institutes for help and eventually gained access to some fancy gpu, allowing us to train larger models in reasonable time.

## Test result (60 sec)

For the test result, we observe there’s a drop at version 1.2.0 when we modified the scanning strategy to combine areas 2 and 3 and also increased the camera's distance from the area surface.

The idea was good, but the model at that point couldn’t handle the blur target well enough.

So we expanded the dataset and upgraded the model to better handle the new conditions. Our average score increased again in later versions.

In our final version 1.3.5, we achieved an average score of 286.8 points, with 99% detection precision across 30 simulation runs.

And that’s everything I wanted to share.
Thank you for listening.
