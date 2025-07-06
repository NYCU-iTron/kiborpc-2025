# Presentation Script

## Introduction (25 sec)

Hello everyone, thank you for being here today.
I’m 林穎沛 from Itron Robotics Team at Chiao Tung University.

Today, I’ll walk you through how we built our program for the competition, including code structure, image pipeline, the challenges we faced and finally, the results we achieved in the simulation.

## Code structure (50 sec)

Our program is modular and organized into two layers, making each component focused and easy to maintain.

In the top layer, we have YourService, which is the default entry point of the framework, and MainControl, where the main task logic is executed.

In the lower layer, we have ItemManager, which stores the detected items, and VisionHandler, which processes camera input and applies yolo models.  

I’ll go over the pipeline details here in the next slide.

We also have Navigator, which handles localization and navigates the robot to the target pose.

In addition, we defined custom data structures like Pose and Item to store information in a consistent format.

## Image pipeline (60 sec)

Here’s how our image pipeline works.

The camera handler first captures a raw image and undistort it using calibration parameters.

Then, the artag detector extracts the tag information and crops the image.

Next, the cropped image goes through preprocessing, such as resizing and normalization, to make sure the image is in the correct format to the model.

After, we feed it to the yolo model. The model will return a list of detected items with their bounding boxes and confidence scores.

In postprocessing, we filter out low confidence result using a threshold.

Finally, we apply weighted box fusion instead of non-maximum suppression to merge overlapping boxes for more accurate results when items are close together.

## Challenge and Solution (60 sec)

We faced some challenges along the way.

First, Programming Mannual has no detail about KiboRpcService class, we couldn’t import the api properly at first. We eventually decompiled the sample APK using Android Studio to understand it and make it work.

Second, the online simulator sometimes failed without any error messages. We had to test line by line to find the bug, which was super slow and frustrating.

Third, training yolo models on laptop is very slow. We reached out to professors and some institutes for help and eventually gained access to some fancy gpu, allowing us to train larger models in a reasonable time.

## Test result (60 sec)

In our final version 1.3.5, we achieved an average score of 286.8 points, with 99% detection precision across 30 simulation runs.

And that’s everything I wanted to share.
Thank you for listening.
