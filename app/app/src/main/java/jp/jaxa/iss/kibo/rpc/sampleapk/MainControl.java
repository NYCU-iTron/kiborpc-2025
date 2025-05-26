package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.types.Pose;

import java.util.List;

import org.opencv.core.Mat;

import android.util.Log;
import android.content.Context;

/**
 * @brief MainControl class for planning the mission.
 * 
 * @param context Context reference.
 * @param api API reference.
 * 
 * Example of using the MainControl class:
 * @code
 * MainControl mainControl = new MainControl(getApplicationContext(), api);
 * mainControl.method1();
 * @endcode
 */
public class MainControl {
    private final String TAG = this.getClass().getSimpleName();

    private Context context;
    private KiboRpcApi api;
    private Navigator navigator;
    private VisionHandler visionHandler;
    private ItemManager itemManager;

    // Assuming CAMERA_TO_ROBOT_TRANSLATION is the vector from robot_center to camera_center in robot_body_frame
    // Assuming CAMERA_TO_ROBOT_ROTATION is the rotation from robot_body_frame to camera_frame (Q_camera_robotbody)
    // For NavCam on Astrobee: (values from Astrobee documentation or calibration)
    // Translation: [0.1102, -0.0421, -0.0826] (X, Y, Z in meters in robot body frame)
    // Rotation: (roll, pitch, yaw) -> (0, -90deg, -90deg) -> Quaternion
    // X_cam = -Z_robot, Y_cam = -X_robot, Z_cam = -Y_robot
    // If robot body frame is (X->fwd, Y->left, Z->up)
    // And NavCam frame is (Z->fwd, X->right, Y->down)
    // Rotation Q_navcam_body : maps vector from body to navcam
    // Body to NavCam: R_z(-90) * R_y(-90)
    // q_body_to_navcam = from_euler(0, -pi/2, -pi/2) = (w:0, x: -0.7071, y: 0, z: -0.7071)
    // We need Q_robotbody_camera (rotation that transforms points from camera frame to robot body frame)
    // Or, Q_camera_robotbody (rotation that transforms points from robot body frame to camera frame)

    // Let's define CAMERA_TO_ROBOT_ROTATION as Q_robot_camera (transforms camera coordinates to robot body coordinates)
    // If NavCam has Z forward, X right, Y down
    // And Robot body has X forward, Y left, Z up
    // To transform a point from NavCam frame to Robot Body frame:
    // P_robot.x =  P_navcam.y
    // P_robot.y = -P_navcam.x
    // P_robot.z = -P_navcam.z
    // This corresponds to a rotation. Let's use a common Astrobee convention if available.
    // For now, using the example values as placeholders.
    // This quaternion transforms a vector from the camera frame to the robot body frame.
    private final Quaternion CAMERA_FRAME_TO_ROBOT_BODY_FRAME_ROTATION = new Quaternion(0f, 0f, -0.7071068f, 0.7071068f); // Example: q_body_navcam.inverse() or q_navcam_body
    // This vector is from the robot's origin (body center) to the camera's origin, expressed in the robot body frame.
    private final Point CAMERA_TRANSLATION_IN_ROBOT_BODY_FRAME = new Point(0.1102, -0.0421, -0.0826);

    public MainControl(Context context, KiboRpcApi api) {
        this.context = context;
        this.api = api;
        this.navigator = new Navigator(api);
        this.visionHandler = new VisionHandler(context, api);
        this.itemManager = new ItemManager(api);
    }

    /**
     * @brief Main method to control the mission.
     * 
     * This method is called when the mission starts.
     * It will call other methods to explore all areas, meet astronauts, and find treasures.
     * 
     * @note This method is intended to be called once with no following code.
     * 
     * @todo maybe a more optimal pose can be used to take the picture of the treasure?
     */
    public void method1() {
        api.startMission();

        // Explore all areas
        for(int areaId = 1; areaId <= 4; areaId++) {
            handleSingleArea(areaId);
        }

        Item treasureItem = meetAstronaut();
        findAndCaptureTreasure(treasureItem);
    }

    public void method2() {
        api.startMission();

        // Explore all areas
        handleSingleArea(1);
        handleCombinedArea();
        handleSingleArea(4);

        Item treasureItem = meetAstronaut();
        findAndCaptureTreasure(treasureItem);
    }

    private void handleSingleArea(int areaId) {
        navigator.navigateToArea(areaId);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        List<Item> areaItems = null;

        int retryMax = 5;
        for (int retry = 1; retry <= retryMax; retry++) {
            Log.i(TAG, "Exploring area " + areaId + " (try " + retry + ")");
            areaItems = visionHandler.inspectArea(areaId);

            if (containsLandmark(areaItems)) break;

            Log.w(TAG, "No landmark found, pause system for a short time then retry.");
            try {
                Thread.sleep(200);
            } catch (Exception e) {
                Log.w(TAG, "Fail to sleep thread" + e);
            }
        }

        if (areaItems == null || !containsLandmark(areaItems)) {
            Log.w(TAG, "No landmark found after retries, leaving to fate.");
            areaItems = visionHandler.guessResult(areaId);
            for (Item item : areaItems) {
                Log.i(TAG, "Item: " + item.getItemId() + ", " + item.getItemName());
            }
        }

        // Treasure Item
        Item treasureItem = areaItems.get(0); 
        if (treasureItem.getItemId() / 10 == 1) {
            itemManager.storeTreasureInfo(treasureItem);
            Log.i(TAG, "Area " + areaId + ": Found treasure " + treasureItem.getItemName());
        } else {
            Log.w(TAG, "Area " + areaId + ": No treasure.");
        }
        
        // Landmark Item
        Item landmarkItem = areaItems.get(1);
        if (landmarkItem.getItemId() / 10 == 2) {
            itemManager.setAreaInfo(landmarkItem);
            Log.i(TAG, "Area " + areaId + ": Found landmark " + landmarkItem.getItemName());
        } else {
            // Should never end up here
            Log.w(TAG, "Area " + areaId + ": No landmark.");
        }
    }

    private void handleCombinedArea() {
        navigator.navigateToArea(5);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        List<Item> area2Items = null;
        List<Item> area3Items = null;
        
        boolean success2 = false;
        boolean success3 = false;

        int retryMax = 5;
        for (int retry = 1; retry <= retryMax; retry++) {
            Log.i(TAG, "Exploring area 2 and 3 together (try " + retry + ")");
            
            if (success2 == true && success3 == true) break;

            if(success2 == false)
                area2Items = visionHandler.inspectArea(2);
            if(success3 == false)
                area3Items = visionHandler.inspectArea(3);
            
            // Check area 2 items
            if (success2 == false && containsLandmark(area2Items)) {
                // Treasure Item
                Item treasureItem = area2Items.get(0);
                if (treasureItem.getItemId() / 10 == 1) {
                    itemManager.storeTreasureInfo(treasureItem);
                    Log.i(TAG, "Area 2: Found treasure " + treasureItem.getItemName());
                } else {
                    Log.w(TAG, "Area 2: No treasure.");
                }
                
                // Landmark Item
                Item landmarkItem = area2Items.get(1);
                if (landmarkItem.getItemId() / 10 == 2) {
                    itemManager.setAreaInfo(landmarkItem);
                    Log.i(TAG, "Area 2: Found landmark " + landmarkItem.getItemName());
                } else {
                    Log.w(TAG, "Area 2: No landmark.");
                }

                success2 = true;
            }

            // Check area 3 items
            if (success3 == false && containsLandmark(area3Items)) {
                // Treasure Item
                Item treasureItem = area3Items.get(0); 
                if (treasureItem.getItemId() / 10 == 1) {
                    itemManager.storeTreasureInfo(treasureItem);
                    Log.i(TAG, "Area 3: Found treasure " + treasureItem.getItemName());
                } else {
                    Log.w(TAG, "Area 3: No treasure.");
                }
                
                // Landmark Item
                Item landmarkItem = area3Items.get(1);
                if (landmarkItem.getItemId() / 10 == 2) {
                    itemManager.setAreaInfo(landmarkItem);
                    Log.i(TAG, "Area 3: Found landmark " + landmarkItem.getItemName());
                } else {
                    // Should never end up here
                    Log.w(TAG, "Area 3: No landmark.");
                }

                success3 = true;               
            }

            Log.w(TAG, "Exploration not completed, pause system for a short time then retry.");

            try {
                Thread.sleep(200);
            } catch (Exception e) {
                Log.w(TAG, "Fail to sleep thread" + e);
            }
        }

        if(success2 == false) {
            handleSingleArea(2);
        }
        if(success3 == false) {
            handleSingleArea(3);
        }
    }

    /**
     * @brief Second part of the mission to meet astronauts.
     */
    private Item meetAstronaut() {
        int areaId = 0;

        // See the real treasure
        navigator.navigateToArea(areaId);
        api.reportRoundingCompletion();
        
        // Recognize the treasure
        List<Item> areaItems = null;
        Item treasureItem = null;

        int retryMax = 20;
        for (int retry = 1; retry <= retryMax; retry++) {
            areaItems = visionHandler.inspectArea(areaId);
            if (containsTreasure(areaItems)) {
                treasureItem = areaItems.get(0);
                break;
            } else {
                Log.w(TAG, "No treasure found, pause system for a short time then retry.");
                    
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Log.w(TAG, "Fail to sleep thread" + e);
                }

                areaItems = visionHandler.inspectArea(areaId);
            }
            
        }

        if (treasureItem == null) {
            Log.w(TAG, "No treasure found, leaving to fate.");
            areaItems = visionHandler.guessResult(areaId);
            treasureItem = areaItems.get(0);
        }

        Log.i(TAG, "Treasure recognized: " + treasureItem.getItemName());
        api.notifyRecognitionItem();

        // Find the area of the treasure and the treasure item
        treasureItem = itemManager.getTreasureInfo(treasureItem);
        Log.i(TAG, "Treasure area: " + treasureItem.getAreaId());

        return treasureItem;
    }

    /**
     * @brief Third part of the mission to find and capture treasure.
     */
    private void findAndCaptureTreasure(Item treasureItem) {
        navigator.navigateToTreasure(treasureItem);
        Log.i(TAG, "Navigating to treasure " + treasureItem.getItemName() + " at " + treasureItem.getAreaId());
        // Capture the treasure image
        visionHandler.captureTreasureImage();
    }

    private boolean containsLandmark(List<Item> items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 2) return true;
        }
        return false;
    }

    private boolean containsTreasure(List<Item> items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 1) return true;
        }
        return false;
    }

    /**
     * Converts AR Tag pose (relative to camera) to map center point (world coordinates).
     *
     * @param arTagPoseInCamera AR Tag pose relative to the camera frame.
     * @param cameraPoseInWorld Camera pose in world coordinates (camera_origin_in_world, Q_world_camera).
     * @return Point representing the AR Tag's center in world coordinates.
     */
    public Point convertARTagToMapCenter(Pose arTagPoseInCamera, Pose cameraPoseInWorld) {
        Point arTagPointInCamera = arTagPoseInCamera.getPoint(); // P_artag_in_camera
        Quaternion cameraOrientationInWorld = cameraPoseInWorld.getQuaternion(); // Q_world_camera
        Point cameraPositionInWorld = cameraPoseInWorld.getPoint();     // C_world (Camera origin in world)

        // Rotate AR tag's point from camera frame to world frame
        // P_artag_in_world_relative_to_camera_origin = Q_world_camera * P_artag_in_camera * Q_world_camera_conj
        double[] rotatedTagPointInWorldFrame = rotatePointByQuaternion(
                arTagPointInCamera.getX(),
                arTagPointInCamera.getY(),
                arTagPointInCamera.getZ(),
                cameraOrientationInWorld
        );

        // Translate AR tag's point by camera's world position
        // P_artag_world = C_world + P_artag_in_world_relative_to_camera_origin
        double worldX = rotatedTagPointInWorldFrame[0] + cameraPositionInWorld.getX();
        double worldY = rotatedTagPointInWorldFrame[1] + cameraPositionInWorld.getY();
        double worldZ = rotatedTagPointInWorldFrame[2] + cameraPositionInWorld.getZ();

        return new Point(worldX, worldY, worldZ);
    }

    /**
     * Converts camera pose in world coordinates to robot center pose in world coordinates.
     *
     * @param cameraPoseInWorld Camera pose in world (camera_origin_in_world, Q_world_camera).
     * @return Robot center pose in world coordinates (robot_origin_in_world, Q_world_robotbody).
     */
    public Pose convertCameraToRobotCenter(Pose cameraPoseInWorld) {
        Point cameraPositionInWorld = cameraPoseInWorld.getPoint();         // C_world
        Quaternion cameraOrientationInWorld = cameraPoseInWorld.getQuaternion(); // Q_world_camera

        // The translation from robot center to camera center, expressed in robot body frame: T_camera_in_robotbody
        Point t_camera_in_robotbody = CAMERA_TRANSLATION_IN_ROBOT_BODY_FRAME;

        // The robot's orientation in the world is Q_world_robotbody.
        // We have Q_world_camera and Q_robotbody_camera (CAMERA_FRAME_TO_ROBOT_BODY_FRAME_ROTATION.conjugate() because we defined it as camera to robot body).
        // Q_world_robotbody = Q_world_camera * Q_camera_robotbody
        // Q_camera_robotbody is the inverse of CAMERA_FRAME_TO_ROBOT_BODY_FRAME_ROTATION if that one is Q_robotbody_camera
        // Let's assume CAMERA_FRAME_TO_ROBOT_BODY_FRAME_ROTATION is Q_robotbody_camera (rotates from camera frame to robot body frame)

        Quaternion q_robotbody_camera = CAMERA_FRAME_TO_ROBOT_BODY_FRAME_ROTATION; // Transforms points from Camera to Robot Body
        Quaternion robotOrientationInWorld = multiplyQuaternions(cameraOrientationInWorld, q_robotbody_camera.conjugate()); // Q_world_robotbody = Q_world_camera * Q_camera_robotbody
                                                                                                                          // Q_camera_robotbody = (Q_robotbody_camera)^-1

        // To find the robot's center position in the world:
        // RobotCenter_world = CameraCenter_world - (Q_world_robotbody * T_camera_in_robotbody * Q_world_robotbody_conj)
        // This is subtracting the camera offset (which is in robot body frame) but rotated to the world frame.

        double[] t_camera_in_world = rotatePointByQuaternion(
            t_camera_in_robotbody.getX(),
            t_camera_in_robotbody.getY(),
            t_camera_in_robotbody.getZ(),
            robotOrientationInWorld // Rotate the T_camera_in_robotbody by the robot's world orientation
        );

        Point robotPositionInWorld = new Point(
            cameraPositionInWorld.getX() - t_camera_in_world[0],
            cameraPositionInWorld.getY() - t_camera_in_world[1],
            cameraPositionInWorld.getZ() - t_camera_in_world[2]
        );

        return new Pose(robotPositionInWorld, robotOrientationInWorld);
    }

    private double[] rotatePointByQuaternion(double px, double py, double pz, Quaternion q_rotation) {
        // P_rotated = q_rotation * P_original * q_rotation_conjugate
        // P_original is (0, px, py, pz)
        Quaternion p_quat = new Quaternion((float)px, (float)py, (float)pz, 0f); // x, y, z, w - Astrobee convention for point quaternion is w=0
        Quaternion q_conj = q_rotation.conjugate();

        // temp = q_rotation * p_quat
        Quaternion temp = multiplyQuaternions(q_rotation, p_quat);
        // P_rotated_quat = temp * q_conj
        Quaternion rotated_p_quat = multiplyQuaternions(temp, q_conj);

        return new double[]{rotated_p_quat.getX(), rotated_p_quat.getY(), rotated_p_quat.getZ()};
    }

    private Quaternion multiplyQuaternions(Quaternion q1, Quaternion q2) {
        // (x1, y1, z1, w1) * (x2, y2, z2, w2)
        float x1 = q1.getX(); float y1 = q1.getY(); float z1 = q1.getZ(); float w1 = q1.getW();
        float x2 = q2.getX(); float y2 = q2.getY(); float z2 = q2.getZ(); float w2 = q2.getW();

        float res_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        float res_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        float res_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
        float res_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;

        return new Quaternion(res_x, res_y, res_z, res_w);
    }
}