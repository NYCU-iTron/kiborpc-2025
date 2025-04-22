# KiboRpcService and KiboRpcApi

## 1. Introduction

`KiboRpcService` is a class that provides a way to communicate with the robot. However, the programming manual only refers some of the methods of this class without providing any other details. For example, you can see that `app/app/src/main/java/jp/jaxa/iss/kibo/rpc/sampleapk/YourService.java` uses `api` directly:

```java
/**
     * @brief The official example of runPlan1() method.
     * 
     * This example is copied from the official example.
     * If you want to use this method, please write this code in the runPlan1() method.
     * 
     * @note This example is copied from runPlan1() method to make the method available for testing.
     */
    private void exampleRunPlan1() {
        // The mission starts.
        api.startMission();

        // ...
```

But the type of `api` is not defined in the file. This note includes the decompiled code of `KiboRpcService` and `KiboRpcApi` classes, retrieved using android studio.

## 2. Decompiled Code

## 2.1. KiboRpcService

```java
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package jp.jaxa.iss.kibo.rpc.api;

import android.util.Log;
import gov.nasa.arc.astrobee.Robot;
import gov.nasa.arc.astrobee.android.gs.MessageType;
import gov.nasa.arc.astrobee.android.gs.StartGuestScienceService;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;
import org.xbill.DNS.ResolverConfig;

public class KiboRpcService extends StartGuestScienceService {
    protected KiboRpcApi api = null;
    public Robot robot;
    public GetterNode getterNode;
    private final DateFormat df = new SimpleDateFormat("yyyyMMdd hhmmssSSS");

    public KiboRpcService() {
    }

    public final void onGuestScienceCustomCmd(String command) {
        this.sendReceivedCustomCommand("info");

        try {
            JSONObject jCommand = new JSONObject(command);
            String sCommand = jCommand.getString("name");
            new JSONObject();
            byte var7 = -1;
            switch(sCommand.hashCode()) {
            case 802033245:
                if (sCommand.equals("runPlan1")) {
                    var7 = 0;
                }
                break;
            case 802033246:
                if (sCommand.equals("runPlan2")) {
                    var7 = 1;
                }
                break;
            case 802033247:
                if (sCommand.equals("runPlan3")) {
                    var7 = 2;
                }
            }

            JSONObject data;
            switch(var7) {
            case 0:
                try {
                    this.runPlan1();
                } catch (Exception var12) {
                    data = new JSONObject();
                    data.put("error", "Program Down: Exception: " + var12.getClass().getName());
                    this.sendData(MessageType.JSON, "data", data.toString());
                    Log.e("KiboRpcApi", "Program Down: Exception: ", var12);
                }

                return;
            case 1:
                try {
                    this.runPlan2();
                } catch (Exception var11) {
                    data = new JSONObject();
                    data.put("error", "Program Down: Exception: " + var11.getClass().getName());
                    this.sendData(MessageType.JSON, "data", data.toString());
                    Log.e("KiboRpcApi", "Program Down: Exception: ", var11);
                }

                return;
            case 2:
                try {
                    this.runPlan3();
                } catch (Exception var10) {
                    data = new JSONObject();
                    data.put("error", "Program Down: Exception: " + var10.getClass().getName());
                    this.sendData(MessageType.JSON, "data", data.toString());
                    Log.e("KiboRpcApi", "Program Down: Exception: ", var10);
                }

                return;
            default:
                this.sendData(MessageType.JSON, "data", "ERROR: Unrecognized command");
                return;
            }
        } catch (JSONException var13) {
            this.sendData(MessageType.JSON, "data", var13.getClass().getName());
            Log.e("KiboRpcApi", "Program Down: Exception: ", var13);
        } catch (Exception var14) {
            this.sendData(MessageType.JSON, "data", var14.getClass().getName());
            Log.e("KiboRpcApi", "Program Down: Exception: ", var14);
        }

    }

    public final void onGuestScienceStart() {
        System.setProperty("dns.server", "127.0.0.1");
        System.setProperty("dns.search", "iss");
        ResolverConfig.refresh();
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCv", "Unable to load OpenCV");
        } else {
            Log.d("OpenCv", "OpenCV loaded");
        }

        this.api = KiboRpcApi.getInstance(this);
        this.sendStarted("info");
    }

    public final void onGuestScienceStop() {
        this.api.shutdownFactory();
        this.sendStopped("info");
        this.terminate();
    }

    protected void runPlan1() {
    }

    protected void runPlan2() {
    }

    protected void runPlan3() {
    }
}
```

## 2.2. KiboRpcApi

```java
//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package jp.jaxa.iss.kibo.rpc.api;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.Bitmap.Config;
import android.util.Log;
import gov.nasa.arc.astrobee.AstrobeeException;
import gov.nasa.arc.astrobee.AstrobeeRuntimeException;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.PendingResult;
import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.Robot;
import gov.nasa.arc.astrobee.RobotFactory;
import gov.nasa.arc.astrobee.Kinematics.Confidence;
import gov.nasa.arc.astrobee.android.gs.MessageType;
import gov.nasa.arc.astrobee.android.gs.StartGuestScienceService;
import gov.nasa.arc.astrobee.ros.DefaultRobotFactory;
import gov.nasa.arc.astrobee.ros.RobotConfiguration;
import gov.nasa.arc.astrobee.types.FlashlightLocation;
import gov.nasa.arc.astrobee.types.FlightMode;
import gov.nasa.arc.astrobee.types.PlannerType;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import jp.jaxa.iss.kibo.rpc.api.areas.AreaItemMap;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

public final class KiboRpcApi extends Activity {
    private GetterNode getterNode;
    private SetterNode setterNode;
    private static final URI ROS_MASTER_URI = URI.create("http://llp:11311");
    private static final String EMULATOR_ROS_HOSTNAME = "hlp";
    private static final String NODE_NAME = "kibo_rpc_api";
    private static final String TARGET_IMAGES_SAVE_DIR = "/immediate/JudgeImages";
    private static final String DEBUG_IMAGES_SAVE_DIR = "/immediate/DebugImages";
    private static KiboRpcApi instance = null;
    private RobotConfiguration robotConfiguration = new RobotConfiguration();
    private RobotFactory factory;
    private Robot robot;
    private StartGuestScienceService gsService = null;
    private PlannerType plannerType = null;
    private final DateFormat df = new SimpleDateFormat("yyyyMMdd hhmmssSSS");
    private final DateFormat dfGametime = new SimpleDateFormat("mmssSSS");
    private final String PAYLOAD_START = "Mission Start";
    private final String PAYLOAD_FINISH = "Mission Finish";
    private final String UNDOCK_START = "Undock Start";
    private final String UNDOCK_FINISH = "Undock Finish";
    private final String UNDOCK_ERROR = "Undock failed";
    private final String SIGNAL_LIGHT_PERCHING_ARM = "Turn on the signal light and deploy the perching arm";
    private final String TARGET_RECOGNITION_COMPLETE = "Target Item recognition is complete";
    private final String TURN_ON_FOUND_PATTERN = "Turn on the found pattern";
    private final String TURN_ON_RECOGNITION_PATTERN = "Turn on the take a snapshot pattern";
    private final String TURN_ON_TAKE_SNAPSHOT_PATTERN = "Turn on the recognition pattern";
    private final String ROUNDING_COMPLETE = "rounding is complete";
    private final String AAR_VERSION = "6.0.0";
    private final int APPROACH_TIMES_SNAPSHOT_SIMULATION = 1;
    private final int APPROACH_TIMES_SNAPSHOT_ISS = 2;
    private final int APPROACH_INTEVAL = 1000;
    private final int LASER_WAIT_TIME = 2000;
    private final int TIMER_CHECK_RESOLUTION = 100;
    private final int UNDOCK_TIMEOUT_ON_SIM = 100;
    private final int MAX_IMAGES = 50;
    private final int MAX_IMAGE_SIZE = 1228800;
    private final int BLINK_NUM = 2;
    private final int WAIT_AFTER_BLINK = 1000;
    private final float FLASH_LIGHT_MAX_IN_FINAL = 0.01F;
    private final float FLASH_LIGHT_MIN = 0.0F;
    private final float FLASH_LIGHT_REPORT_ROUNDING_COMPLETION = 0.05F;
    private final boolean LIGHT_MODE_FRONT = true;
    private final boolean LIGHT_MODE_BACK = false;
    private final double[] NAVCAM_CAMERA_MATRIX_SIMULATION = new double[]{523.10575D, 0.0D, 635.434258D, 0.0D, 534.765913D, 500.335102D, 0.0D, 0.0D, 1.0D};
    private final double[] NAVCAM_CAMERA_MATRIX_ISS = new double[]{608.8073D, 0.0D, 632.53684D, 0.0D, 607.61439D, 549.08386D, 0.0D, 0.0D, 1.0D};
    private final double[] NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION = new double[]{-0.164787D, 0.020375D, -0.001572D, -3.69E-4D, 0.0D};
    private final double[] NAVCAM_DISTORTION_COEFFICIENTS_ISS = new double[]{-0.212191D, 0.073843D, -9.18E-4D, 0.00189D, 0.0D};
    private final double[] DOCKCAM_CAMERA_MATRIX_SIMULATION = new double[]{661.783002D, 0.0D, 595.212041D, 0.0D, 671.508662D, 489.094196D, 0.0D, 0.0D, 1.0D};
    private final double[] DOCKCAM_CAMERA_MATRIX_ISS = new double[]{753.51021D, 0.0D, 631.11512D, 0.0D, 751.3611D, 508.69621D, 0.0D, 0.0D, 1.0D};
    private final double[] DOCKCAM_DISTORTION_COEFFICIENTS_SIMULATION = new double[]{-0.215168D, 0.044354D, 0.003615D, 0.005093D, 0.0D};
    private final double[] DOCKCAM_DISTORTION_COEFFICIENTS_ISS = new double[]{-0.411405D, 0.17724D, -0.017145D, 0.006421D, 0.0D};
    private final float DISTANCE_THRESHOLD = 0.3F;
    private boolean reportCompletion = false;
    private boolean tookTargetItemSnap = false;
    private AreaItemMap areaItemMap;
    private long startMillis;
    private long finishMillis;

    private void sendExceptionMessage(String errmsg, Exception err) {
        Log.v("KiboRpcApi", "[Start] sendExceptionMessage");

        try {
            JSONObject data = new JSONObject();
            data.put("error", "[" + err.getClass().getName() + "] " + errmsg);
            Log.e("KiboRpcApi", errmsg, err);
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
        } catch (JSONException var4) {
            this.gsService.sendData(MessageType.JSON, "data", var4.getClass().getName());
            Log.e("KiboRpcApi", "[sendExceptionMessage] Failed to send error message.", var4);
        }

        Log.v("KiboRpcApi", "[Finish] sendExceptionMessage");
    }

    private KiboRpcApi(StartGuestScienceService startGuestScienceService) {
        Log.v("KiboRpcApi", "[Start] KiboRpcApi");
        Log.i("KiboRpcApi", "[AARVersion] AAR version 6.0.0");
        this.configureRobot();
        this.factory = new DefaultRobotFactory(this.robotConfiguration);
        this.gsService = startGuestScienceService;

        try {
            this.robot = this.factory.getRobot();
            this.getterNode = GetterNode.getInstance();
            this.setterNode = SetterNode.getInstance();
            this.areaItemMap = new AreaItemMap();
        } catch (AstrobeeException var3) {
            this.sendExceptionMessage("[Constructor] Error with Astrobee", var3);
        } catch (InterruptedException var4) {
            this.sendExceptionMessage("[Constructor] Connection Interrupted", var4);
        }

        Log.v("KiboRpcApi", "[Finish] KiboRpcApi");
    }

    private void configureRobot() {
        Log.v("KiboRpcApi", "[Start] configureRobot");
        this.robotConfiguration.setMasterUri(ROS_MASTER_URI);
        this.robotConfiguration.setHostname("hlp");
        this.robotConfiguration.setNodeName("kibo_rpc_api");
        Log.v("KiboRpcApi", "[Finish] configureRobot");
    }

    private Result getCommandResult(PendingResult pending, boolean printRobotPosition, int timeout) {
        Log.v("KiboRpcApi", "[Start] getCommandResult");
        Result result = null;
        int counter = 0;

        try {
            label100: {
                while(!pending.isFinished()) {
                    if (timeout >= 0) {
                        Log.v("KiboRpcApi", "[getCommandResult] Setting timeout");
                        if (counter > timeout) {
                            Log.v("KiboRpcApi", "[getCommandResult] return null");
                            Object var7 = null;
                        }
                    }

                    if (printRobotPosition) {
                        Log.v("KiboRpcApi", "[getCommandResult] Meanwhile, let's get the positioning along the trajectory");
                        Kinematics k = this.getterNode.getCurrentKinematics();
                        if (k.getPosition() != null) {
                            Log.i("KiboRpcApi", "[getCommandResult] Current Position: " + k.getPosition().toString());
                        }

                        if (k.getOrientation() != null) {
                            Log.i("KiboRpcApi", "[getCommandResult] Current Orientation: " + k.getOrientation().toString());
                        }
                    }

                    pending.getResult(1000L, TimeUnit.MILLISECONDS);
                    ++counter;
                }

                result = pending.getResult();
                this.printLogCommandResult(result);
            }
        } catch (AstrobeeException var13) {
            this.sendExceptionMessage("[getCommandResult] Error with Astrobee", var13);
        } catch (InterruptedException var14) {
            this.sendExceptionMessage("[getCommandResult] Connection Interrupted", var14);
        } catch (TimeoutException var15) {
            this.sendExceptionMessage("[getCommandResult] Timeout connection", var15);
        } finally {
            Log.v("KiboRpcApi", "[Finish] getCommandResult");
            return result;
        }
    }

    private void printLogCommandResult(Result result) {
        Log.i("KiboRpcApi", "[Start] printLogCommandResult");
        if (result != null) {
            if (result.getStatus() != null) {
                Log.i("KiboRpcApi", "[printLogCommandResult] Command status: " + result.getStatus().toString());
            }

            if (!result.hasSucceeded()) {
                Log.e("KiboRpcApi", "[printLogCommandResult] Command message: " + result.getMessage());
            }

            Log.i("KiboRpcApi", "[printLogCommandResult] Done.");
            Log.i("KiboRpcApi", "[Finish] printLogCommandResult");
        } else {
            Log.e("KiboRpcApi", "[printLogCommandResult] Invalid result.");
        }

    }

    private Result stopAllMotion() {
        Log.v("KiboRpcApi", "[Start] stopAllMotion");
        PendingResult pendingResult = null;

        try {
            pendingResult = this.robot.stopAllMotion();
        } catch (AstrobeeRuntimeException var3) {
            this.sendExceptionMessage("[stopAllMotion] Node not ready or dead.", var3);
            return null;
        }

        Result result = this.getCommandResult(pendingResult, false, -1);
        Log.v("KiboRpcApi", "[Finish] stopAllMotion");
        return result;
    }

    private boolean setPlanner(PlannerType plannerType) {
        Log.v("KiboRpcApi", "[Start] setPlanner");
        PendingResult pendingResult = null;

        try {
            pendingResult = this.robot.setPlanner(plannerType);
        } catch (AstrobeeRuntimeException var4) {
            this.sendExceptionMessage("[setPlanner] Node not ready or dead.", var4);
            return false;
        }

        Result result = this.getCommandResult(pendingResult, false, 5);
        if (result != null) {
            if (result.hasSucceeded()) {
                this.plannerType = plannerType;
                Log.v("KiboRpcApi", "[setPlanner] Planner is set to " + plannerType);
            }

            Log.v("KiboRpcApi", "[Finish] setPlanner");
            return result.hasSucceeded();
        } else {
            Log.e("KiboRpcApi", "[setPlanner] Invalid result.");
            return false;
        }
    }

    public void shutdownFactory() {
        Log.v("KiboRpcApi", "[Start] shutdownFactory");
        this.factory.shutdown();
        Log.v("KiboRpcApi", "[Finish] shutdownFactory");
    }

    public static KiboRpcApi getInstance(StartGuestScienceService startGuestScienceService) {
        Log.v("KiboRpcApi", "[Start] getInstance");
        if (instance == null) {
            Log.v("KiboRpcApi", "[getInstance] Make instance");
            instance = new KiboRpcApi(startGuestScienceService);
        }

        Log.v("KiboRpcApi", "[Finish] getInstance");
        return instance;
    }

    private Kinematics getTrustedRobotKinematics() {
        Log.i("KiboRpcApi", "[Start] getTrustedRobotKinematics");
        Log.i("KiboRpcApi", "[getTrustedRobotKinematics] Waiting for robot to acquire position.");
        Kinematics k = null;
        long start_point = System.currentTimeMillis();

        for(long end_point = System.currentTimeMillis(); end_point - start_point < 30000L; end_point = System.currentTimeMillis()) {
            k = this.getterNode.getCurrentKinematics();
            if (k.getConfidence() == Confidence.GOOD) {
                Log.v("KiboRpcApi", "[getTrustedRobotKinematics] Break loop");
                break;
            }

            k = null;

            try {
                Thread.sleep(1000L);
            } catch (InterruptedException var7) {
                Log.e("KiboRpcApi", "[getTrustedRobotkinematics] It was not possible to get a trusted kinematics. Sorry.");
                return null;
            }
        }

        Log.i("KiboRpcApi", "[Finish] getTrustedRobotKinematics");
        return k;
    }

    public Kinematics getRobotKinematics() {
        Log.i("KiboRpcApi", "[Start] getRobotKinematics");
        Log.i("KiboRpcApi", "[getRobotKinematics] Waiting for robot to acquire position.");
        Log.i("KiboRpcApi", "[Finish] getRobotKinematics");
        return this.getterNode.getCurrentKinematics();
    }

    public Bitmap getBitmapNavCam() {
        Log.i("KiboRpcApi", "[Start] getBitmapNavCam");
        Bitmap ret = this.getterNode.getBitmapNavCam();
        if (ret == null) {
            Log.i("KiboRpcApi", "[getBitmapNavCam] ret is null.");
            Log.e("KiboRpcApi", "[getBitmapNavCam] It was not possible to get a Bitmap from Nav Cam.");
            return ret;
        } else {
            if (!this.getterNode.getOnSimulation()) {
                Log.v("KiboRpcApi", "[getBitmapNavCam] getOnSimulation: False");
                Date mDate = new Date();
                SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMddHHmmss");
                String fileName = "NavCam_Bitmap_Snapshot_" + fileNameDate.format(mDate) + ".jpg";
                this.saveBitmapImage(ret, fileName);
                this.setSignalStateTakeSnapshotPattern();
                this.setterNode.setSignalState((byte)17, 1);
            }

            Log.i("KiboRpcApi", "[Finish] getBitmapNavCam");
            return ret;
        }
    }

    public Bitmap getBitmapDockCam() {
        Log.i("KiboRpcApi", "[Start] getBitmapDockCam");
        Bitmap ret = this.getterNode.getBitmapDockCam();
        if (ret == null) {
            Log.i("KiboRpcApi", "[getBitmapDockCam] ret is null.");
            Log.e("KiboRpcApi", "[getBitmapDockCam] It was not possible to get a Bitmap from Dock Cam.");
            return ret;
        } else {
            if (!this.getterNode.getOnSimulation()) {
                Log.v("KiboRpcApi", "[getBitmapDockCam] getOnSimulation: False");
                Date mDate = new Date();
                SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMddHHmmss");
                String fileName = "DockCam_Bitmap_Snapshot_" + fileNameDate.format(mDate) + ".jpg";
                this.saveBitmapImage(ret, fileName);
                this.setSignalStateTakeSnapshotPattern();
                this.setterNode.setSignalState((byte)17, 1);
            }

            Log.i("KiboRpcApi", "[Finish] getBitmapDockCam");
            return ret;
        }
    }

    public Mat getMatNavCam() {
        Log.i("KiboRpcApi", "[Start] getMatNavCam");
        Mat ret = this.getterNode.getMatNavCam();
        if (ret == null) {
            Log.i("KiboRpcApi", "[getMatNavCam] ret is null.");
            Log.e("KiboRpcApi", "[getMatNavCam] It was not possible to get a Mat from Nav Cam.");
            return ret;
        } else {
            if (!this.getterNode.getOnSimulation()) {
                Log.v("KiboRpcApi", "[getMatNavCam] getOnSimulation: False");
                Date mDate = new Date();
                SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMddHHmmss");
                String fileName = "NavCam_Mat_Snapshot_" + fileNameDate.format(mDate) + ".jpg";
                this.saveMatImage(ret, fileName);
                this.setSignalStateTakeSnapshotPattern();
                this.setterNode.setSignalState((byte)17, 1);
            }

            Log.i("KiboRpcApi", "[Finish] getMatNavCam");
            return ret;
        }
    }

    public Mat getMatDockCam() {
        Log.i("KiboRpcApi", "[Start] getMatDockCam");
        Mat ret = this.getterNode.getMatDockCam();
        if (ret == null) {
            Log.i("KiboRpcApi", "[getMatDockCam] ret is null.");
            Log.e("KiboRpcApi", "[getMatDockCam] It was not possible to get a Mat from Dock Cam.");
            return ret;
        } else {
            if (!this.getterNode.getOnSimulation()) {
                Log.v("KiboRpcApi", "[getMatDockCam] getOnSimulation: False");
                Date mDate = new Date();
                SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMddHHmmss");
                String fileName = "DockCam_Mat_Snapshot_" + fileNameDate.format(mDate) + ".jpg";
                this.saveMatImage(ret, fileName);
                this.setSignalStateTakeSnapshotPattern();
                this.setterNode.setSignalState((byte)17, 1);
            }

            Log.i("KiboRpcApi", "[Finish] getMatDockCam");
            return ret;
        }
    }

    public Result flashlightControlFront(float brightness) {
        Log.i("KiboRpcApi", "[Start] flashlightControlFront");
        if (!this.getterNode.getOnSimulation() && brightness > 0.01F) {
            Log.v("KiboRpcApi", "[flashlightControlFront] In the final, the maximum value of the light is set to 0.01.");
            brightness = 0.01F;
        }

        Log.i("KiboRpcApi", "Parameters: brightness == " + brightness);
        PendingResult pendingResult = null;

        try {
            pendingResult = this.robot.setFlashlightBrightness(FlashlightLocation.FRONT, brightness);
        } catch (AstrobeeRuntimeException var4) {
            this.sendExceptionMessage("[flashlightControlFront] Node not ready or dead.", var4);
            return null;
        }

        Log.i("KiboRpcApi", "[Finish] flashlightControlFront");
        return this.getCommandResult(pendingResult, false, -1);
    }

    public Result flashlightControlBack(float brightness) {
        Log.i("KiboRpcApi", "[Start] flashlightControlBack");
        if (!this.getterNode.getOnSimulation() && brightness > 0.01F) {
            Log.v("KiboRpcApi", "[flashlightControlBack] In the final, the maximum value of the light is set to 0.01.");
            brightness = 0.01F;
        }

        Log.i("KiboRpcApi", "Parameters: brightness == " + brightness);
        PendingResult pendingResult = null;

        try {
            pendingResult = this.robot.setFlashlightBrightness(FlashlightLocation.BACK, brightness);
        } catch (AstrobeeRuntimeException var4) {
            this.sendExceptionMessage("[flashlightControlBack] Node not ready or dead.", var4);
            return null;
        }

        Log.i("KiboRpcApi", "[Finish] flashlightControlBack");
        return this.getCommandResult(pendingResult, false, -1);
    }

    private Result flashlightControl(float brightness, boolean front) {
        FlashlightLocation flloc = FlashlightLocation.FRONT;
        if (front) {
            Log.i("KiboRpcApi", "[Start] flashlightControl: front");
            Log.i("KiboRpcApi", "Parameters: brightness == " + brightness);
            flloc = FlashlightLocation.FRONT;
        } else {
            Log.i("KiboRpcApi", "[Start] flashlightControl: back");
            Log.i("KiboRpcApi", "Parameters: brightness == " + brightness);
            flloc = FlashlightLocation.BACK;
        }

        PendingResult pendingResult = null;

        try {
            pendingResult = this.robot.setFlashlightBrightness(flloc, brightness);
        } catch (AstrobeeRuntimeException var6) {
            this.sendExceptionMessage("[flashlightControlFront] Node not ready or dead.", var6);
            return null;
        }

        Log.i("KiboRpcApi", "[Finish] flashlightControlFront");
        return this.getCommandResult(pendingResult, false, -1);
    }

    public Result moveTo(Point goalPoint, Quaternion orientation, boolean printRobotPosition) {
        Log.i("KiboRpcApi", "[Start] moveTo");
        Log.i("KiboRpcApi", "Parameters: goalPoint == " + goalPoint + ", orientation == " + orientation + ", printRobotPosition == " + printRobotPosition);
        if (goalPoint == null) {
            Log.e("KiboRpcApi", "[moveTo] goalPoint is invalid. goalPoint == " + goalPoint);
            return null;
        } else if (orientation == null) {
            Log.e("KiboRpcApi", "[moveTo] orientation is invalid. orientation == " + orientation);
            return null;
        } else if (!this.setPlanner(PlannerType.TRAPEZOIDAL)) {
            Log.e("KiboRpcApi", "[moveTo] Cannot set planner.");
            return null;
        } else {
            Result result = this.stopAllMotion();
            if (result == null) {
                Log.e("KiboRpcApi", "[moveTo] Cannot stop all motion. result == " + result);
                return null;
            } else {
                if (result.hasSucceeded()) {
                    PendingResult pendingResult = null;

                    try {
                        pendingResult = this.robot.simpleMove6DOF(goalPoint, orientation);
                    } catch (AstrobeeRuntimeException var7) {
                        this.sendExceptionMessage("[moveTo] Node not ready or dead.", var7);
                        return null;
                    }

                    result = this.getCommandResult(pendingResult, printRobotPosition, -1);
                }

                Log.i("KiboRpcApi", "[Finish] moveTo");
                return result;
            }
        }
    }

    public Result relativeMoveTo(Point goalPoint, Quaternion orientation, boolean printRobotPosition) {
        Log.i("KiboRpcApi", "[Start] relativeMoveTo");
        Log.i("KiboRpcApi", "Parameters: goalPoint == " + goalPoint + ", orientation == " + orientation + ", printRobotPosition == " + printRobotPosition);
        if (goalPoint == null) {
            Log.e("KiboRpcApi", "[relativeMoveTo] goalPoint is invalid. goalPoint == " + goalPoint);
            return null;
        } else if (orientation == null) {
            Log.e("KiboRpcApi", "[relativeMoveTo] orientation is invalid. orientation == " + orientation);
            return null;
        } else {
            Kinematics k = this.getTrustedRobotKinematics();
            if (k == null) {
                Log.e("KiboRpcApi", "[relativeMoveTo] Cannot get robot kinematics. k == " + k);
                return null;
            } else {
                Point currPosition = k.getPosition();
                Point endPoint = new Point(currPosition.getX() + goalPoint.getX(), currPosition.getY() + goalPoint.getY(), currPosition.getZ() + goalPoint.getZ());
                Log.i("KiboRpcApi", "[Finish] relativeMoveTo");
                return this.moveTo(endPoint, orientation, printRobotPosition);
            }
        }
    }

    private void setSignalStateFoundPattern() {
        try {
            JSONObject data = new JSONObject();
            Log.v("KiboRpcApi", "[Start] setSignalStateFoundPattern");
            this.setterNode.setSignalState((byte)4, 2);
            this.setterNode.setSignalState((byte)5, 1);
            this.setterNode.setSignalState((byte)3, 1);
            data.put("signal_light", "Turn on the found pattern");
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            Log.v("KiboRpcApi", "[Finish] setSignalStateFoundPattern");
        } catch (JSONException var2) {
            this.sendExceptionMessage("[setSignalStateFoundPattern] Internal error occurred. Unable to serialize data to JSON.", var2);
        } catch (Exception var3) {
            this.sendExceptionMessage("[setSignalStateFoundPattern] Internal error occurred. Unable to send data to gds.", var3);
        }

    }

    private void setSignalStateRecognitionPattern() {
        try {
            JSONObject data = new JSONObject();
            Log.v("KiboRpcApi", "[Start] setSignalStateRecognitionPattern");
            this.setterNode.setSignalState((byte)3, 1);
            data.put("signal_light", "Turn on the take a snapshot pattern");
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            Log.v("KiboRpcApi", "[Finish] setSignalStateRecognitionPattern");
        } catch (JSONException var2) {
            this.sendExceptionMessage("[setSignalStateRecognitionPattern] Internal error occurred. Unable to serialize data to JSON.", var2);
        } catch (Exception var3) {
            this.sendExceptionMessage("[setSignalStateRecognitionPattern] Internal error occurred. Unable to send data to gds.", var3);
        }

    }

    private void setSignalStateTakeSnapshotPattern() {
        try {
            JSONObject data = new JSONObject();
            Log.v("KiboRpcApi", "[Start] setSignalStateTakeSnapshotPattern");
            this.setterNode.setSignalState((byte)7, 1);
            data.put("signal_light", "Turn on the recognition pattern");
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            Log.v("KiboRpcApi", "[Finish] setSignalStateTakeSnapshotPattern");
        } catch (JSONException var2) {
            this.sendExceptionMessage("[setSignalStateTakeSnapshotPattern] Internal error occurred. Unable to serialize data to JSON.", var2);
        } catch (Exception var3) {
            this.sendExceptionMessage("[setSignalStateTakeSnapshotPattern] Internal error occurred. Unable to send data to gds.", var3);
        }

    }

    public boolean startMission() {
        Log.i("KiboRpcApi", "[Start] startMission");
        boolean isSuccess = false;

        try {
            JSONObject data = new JSONObject();
            String time = this.df.format(new Date(System.currentTimeMillis()));
            data.put("timestamp", time);
            data.put("status", "Undock Start");
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            Log.i("KiboRpcApi", "Undock Start: " + time);
            PendingResult pendingResult = this.robot.undock();
            int timeout = 100;
            if (!this.getterNode.getOnSimulation()) {
                Log.d("KiboRpcApi", "set timeout -1");
                timeout = -1;
            }

            Result result = this.getCommandResult(pendingResult, false, timeout);
            data = new JSONObject();
            if (result != null && result.hasSucceeded()) {
                data.put("status", "Undock Finish");
                isSuccess = true;
                time = this.df.format(new Date(System.currentTimeMillis()));
                Log.i("KiboRpcApi", "Undock Finish: " + time);
            } else {
                data.put("failed", "Undock failed");
                time = this.df.format(new Date(System.currentTimeMillis()));
                Log.e("KiboRpcApi", "Undock failed: " + time);
            }

            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            if (isSuccess) {
                time = this.df.format(new Date(System.currentTimeMillis()));
                data = new JSONObject();
                data.put("timestamp", time);
                data.put("status", "Mission Start");
                this.gsService.sendData(MessageType.JSON, "data", data.toString());
                this.startMillis = this.df.parse(time).getTime();
                Log.i("KiboRpcApi", "Mission Start: " + time);
            }
        } catch (JSONException var7) {
            this.sendExceptionMessage("[startMission] Internal error occurred. Unable to serialize data to JSON.", var7);
            return false;
        } catch (Exception var8) {
            this.sendExceptionMessage("[startMission] Internal error occurred. Unable to send data to gds.", var8);
            return false;
        }

        Log.i("KiboRpcApi", "[Finish] startMission");
        return isSuccess;
    }

    /** @deprecated */
    @Deprecated
    private Result setQuietMode() {
        try {
            Log.v("KiboRpcApi", "[Start] setQuietMode");
            Log.i("KiboRpcApi", "[setQuietMode] Change the flight mode to Quiet mode.");
            PendingResult pendingResult = this.robot.setOperatingLimits("iss_quiet", FlightMode.QUIET, 0.02F, 0.002F, 0.0174F, 0.0174F, 0.25F);
            Result result = this.getCommandResult(pendingResult, true, -1);
            Log.i("KiboRpcApi", "[setQuietMode] Changed flight mode to Quiet mode.");
            Log.v("KiboRpcApi", "[Finish] setQuietMode");
            return result;
        } catch (AstrobeeRuntimeException var3) {
            this.sendExceptionMessage("[setQuietMode] Node not ready or dead.", var3);
            return null;
        }
    }

    /** @deprecated */
    @Deprecated
    private Result setLomoMode() {
        try {
            Log.v("KiboRpcApi", "[Start] setLomoMode");
            Log.i("KiboRpcApi", "[setLomoMode] Change the flight mode to Lomo mode.");
            PendingResult pendingResult = this.robot.setOperatingLimits("iss_lomo", FlightMode.NOMINAL, 0.15F, 0.0175F, 0.0873F, 0.1745F, 0.25F);
            Result result = this.getCommandResult(pendingResult, true, -1);
            Log.i("KiboRpcApi", "[setLomoMode] Changed flight mode to Lomo mode.");
            Log.v("KiboRpcApi", "[Finish] setLomoMode");
            return result;
        } catch (AstrobeeRuntimeException var3) {
            this.sendExceptionMessage("[setLomoMode] Node not ready or dead.", var3);
            return null;
        }
    }

    public void notifyRecognitionItem() {
        Log.i("KiboRpcApi", "[Start] notifyRecognitionItem");

        try {
            Log.v("KiboRpcApi", "[notifyRecognitionItem] Target Item recognition is complete");
            JSONObject data = new JSONObject();
            data.put("status", "Target Item recognition is complete");
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            if (!this.getterNode.getOnSimulation()) {
                this.setSignalStateRecognitionPattern();
                this.setterNode.setSignalState((byte)17, 1);
            }
        } catch (JSONException var2) {
            this.sendExceptionMessage("[notifyRecognitionItem] Internal error was occurred. Unable to serialize data to JSON.", var2);
            return;
        } catch (Exception var3) {
            this.sendExceptionMessage("[notifyRecognitionItem] Internal error was occurred. Unable to send data to gds.", var3);
            return;
        }

        Log.i("KiboRpcApi", "[Finish] notifyRecognitionItem");
    }

    public boolean reportRoundingCompletion() {
        Log.i("KiboRpcApi", "[Start] reportRoundingCompletion");

        try {
            if (!this.reportCompletion) {
                Log.v("KiboRpcApi", "[reportRoundingCompletion] Make rounding is complete message");
                JSONObject data = new JSONObject();
                data.put("timestamp", this.df.format(new Date(System.currentTimeMillis())));
                data.put("status", "rounding is complete");
                new JSONObject();
                JSONObject areamap = this.areaItemMap.getAreaItemMapJson();
                Iterator jsonkey = areamap.keys();

                String positionX;
                while(jsonkey.hasNext()) {
                    positionX = (String)jsonkey.next();
                    data.put(positionX, areamap.getJSONArray(positionX));
                }

                Log.v("KiboRpcApi", "[reportRoundingCompletion] Do getRobotKinematics");
                Kinematics kinematics = this.getRobotKinematics();
                if (kinematics == null || kinematics.getPosition() == null || kinematics.getOrientation() == null) {
                    Log.e("KiboRpcApi", "[reportRoundingCompletion] It was not possible to get a kinematics.");
                    if (kinematics == null) {
                        Log.e("KiboRpcApi", "[reportRoundingCompletion] kinematics is null.");
                    }

                    if (kinematics.getPosition() == null) {
                        Log.e("KiboRpcApi", "[reportRoundingCompletion] kinematics.getPosition() is null.");
                    }

                    if (kinematics.getOrientation() == null) {
                        Log.e("KiboRpcApi", "[reportRoundingCompletion] kinematics.getOrientation() is null.");
                    }

                    return false;
                }

                positionX = String.format("%.3f", kinematics.getPosition().getX());
                String positionY = String.format("%.3f", kinematics.getPosition().getY());
                String positionZ = String.format("%.3f", kinematics.getPosition().getZ());
                String orientationX = String.format("%.3f", kinematics.getOrientation().getX());
                String orientationY = String.format("%.3f", kinematics.getOrientation().getY());
                String orientationZ = String.format("%.3f", kinematics.getOrientation().getZ());
                String orientationW = String.format("%.3f", kinematics.getOrientation().getW());
                data.put("rounding_point_pos", "[ posX : " + positionX + ", posY : " + positionY + ", posZ " + positionZ + ", quaX: " + orientationX + ", quaY: " + orientationY + ", quaZ: " + orientationZ + ", quaW: " + orientationW + " ]");
                this.gsService.sendData(MessageType.JSON, "data", data.toString());

                for(int i = 0; i < 2; ++i) {
                    Log.v("KiboRpcApi", "[reportRoundingCompletion] Count: " + i);
                    this.flashlightControl(0.05F, true);
                    this.flashlightControl(0.05F, false);
                    this.flashlightControl(0.0F, true);
                    this.flashlightControl(0.0F, false);
                }

                Thread.sleep(1000L);
                this.reportCompletion = true;
            }
        } catch (JSONException var11) {
            this.sendExceptionMessage("[reportRoundingCompletion] Internal error was occurred. Unable to serialize data to JSON.", var11);
            this.reportCompletion = false;
            return false;
        } catch (Exception var12) {
            this.sendExceptionMessage("[reportRoundingCompletion] Internal error was occurred.", var12);
            this.reportCompletion = false;
            return false;
        }

        Log.i("KiboRpcApi", "[Finish] reportRoundingCompletion");
        return true;
    }

    public void takeTargetItemSnapshot() {
        Log.i("KiboRpcApi", "[Start] takeTargetItemSnapshot");

        try {
            if (!this.tookTargetItemSnap) {
                boolean result = true;
                int approach_time = 1;
                if (!this.getterNode.getOnSimulation()) {
                    Log.v("KiboRpcApi", "[takeTargetItemSnapshot] getOnSimulation: False");
                    Log.v("KiboRpcApi", "[takeTargetItemSnapshot] Set approach_time");
                    approach_time = 2;
                }

                result = this.takeSnapshot(approach_time);
                if (!result) {
                    Log.e("KiboRpcApi", "[takeTargetItemSnapshot] Fail to Take the targetItem Snapshot.");
                }

                Log.v("KiboRpcApi", "[takeTargetItemSnapshot] Make finish message");
                JSONObject data = new JSONObject();
                String time = this.df.format(new Date(System.currentTimeMillis()));
                this.finishMillis = this.df.parse(time).getTime();
                long diffMillis = this.finishMillis - this.startMillis;
                String gameTime = this.dfGametime.format(new Date(diffMillis));
                data.put("timestamp", time);
                data.put("mission_time", gameTime);
                data.put("status", "Mission Finish");
                this.gsService.sendData(MessageType.JSON, "data", data.toString());
                Log.v("KiboRpcApi", "[takeTargetItemSnapshot] Do getRobotKinematics");
                Kinematics kinematics = this.getRobotKinematics();
                if (kinematics != null && kinematics.getPosition() != null && kinematics.getOrientation() != null) {
                    String positionX = String.format("%.3f", kinematics.getPosition().getX());
                    String positionY = String.format("%.3f", kinematics.getPosition().getY());
                    String positionZ = String.format("%.3f", kinematics.getPosition().getZ());
                    String orientationX = String.format("%.3f", kinematics.getOrientation().getX());
                    String orientationY = String.format("%.3f", kinematics.getOrientation().getY());
                    String orientationZ = String.format("%.3f", kinematics.getOrientation().getZ());
                    String orientationW = String.format("%.3f", kinematics.getOrientation().getW());
                    data = new JSONObject();
                    data.put("found_point_pos", "[ posX : " + positionX + ", posY : " + positionY + ", posZ " + positionZ + ", quaX: " + orientationX + ", quaY: " + orientationY + ", quaZ: " + orientationZ + ", quaW: " + orientationW + " ]");
                    this.gsService.sendData(MessageType.JSON, "data", data.toString());
                } else {
                    Log.e("KiboRpcApi", "[takeTargetItemSnapshot] It was not possible to get a kinematics.");
                    if (kinematics == null) {
                        Log.e("KiboRpcApi", "[takeTargetItemSnapshot] kinematics is null.");
                    }

                    if (kinematics.getPosition() == null) {
                        Log.e("KiboRpcApi", "[takeTargetItemSnapshot] kinematics.getPosition() is null.");
                    }

                    if (kinematics.getOrientation() == null) {
                        Log.e("KiboRpcApi", "[takeTargetItemSnapshot] kinematics.getOrientation() is null.");
                    }
                }

                if (!this.getterNode.getOnSimulation()) {
                    Log.v("KiboRpcApi", "[takeTargetItemSnapshot] getOnSimulation: False");
                    Log.v("KiboRpcApi", "[takeTargetItemSnapshot] Set SignalState");
                    this.setSignalStateFoundPattern();
                    this.setterNode.setSignalState((byte)17, 1);
                }

                this.tookTargetItemSnap = true;
            }
        } catch (JSONException var16) {
            this.sendExceptionMessage("[takeTargetItemSnapshot] Internal error was occurred. Unable to serialize data to JSON.", var16);
            this.tookTargetItemSnap = false;
            return;
        } catch (Exception var17) {
            this.sendExceptionMessage("[takeTargetItemSnapshot] Internal error was occurred.", var17);
            this.tookTargetItemSnap = false;
            return;
        }

        Log.i("KiboRpcApi", "[Finish] takeTargetItemSnapshot");
    }

    private boolean takeSnapshot(int approach_times) {
        try {
            Log.v("KiboRpcApi", "[Start] takeSnapshot");
            JSONObject data = null;

            for(int i = 1; i <= approach_times; ++i) {
                Log.v("KiboRpcApi", "[takeSnapshot] Count: " + i);
                long startTime = System.currentTimeMillis();
                if (i == 1) {
                    Log.v("KiboRpcApi", "[takeSnapshot] Make start message");
                }

                Log.v("KiboRpcApi", "[takeSnapshot] Make target snapshot");
                if (!this.getterNode.getOnSimulation()) {
                    Log.v("KiboRpcApi", "[takeSnapshot] getOnSimulation: False");
                    Log.v("KiboRpcApi", "[takeSnapshot] Save bitmap");
                    Bitmap image = this.getBitmapNavCam();
                    this.saveBitmap(image);
                }

                for(long currentTime = System.currentTimeMillis(); currentTime - startTime <= 1000L; currentTime = System.currentTimeMillis()) {
                    if (this.getterNode.getOnSimulation()) {
                        Log.v("KiboRpcApi", "[takeSnapshot] getOnSimulation: True");
                        Log.v("KiboRpcApi", "[takeSnapshot] Sleep inteval");
                        Thread.sleep(10L);
                    } else {
                        Log.v("KiboRpcApi", "[takeSnapshot] getOnSimulation: False");
                        Log.v("KiboRpcApi", "[takeSnapshot] Sleep inteval");
                        Thread.sleep(10L);
                    }
                }
            }

            Log.v("KiboRpcApi", "[takeSnapshot] Make target finish message");
            Log.v("KiboRpcApi", "[Finish] takeSnapshot");
            return true;
        } catch (SecurityException var8) {
            this.sendExceptionMessage("[takeSnapshot] Internal error was occurred. Unable to access directory.", var8);
            return false;
        } catch (IOException var9) {
            this.sendExceptionMessage("[takeSnapshot] Internal error was occurred. Unable to make image files.", var9);
            return false;
        } catch (NullPointerException var10) {
            this.sendExceptionMessage("[takeSnapshot] Internal error was occurred. Could not get Nav cam image.", var10);
            return false;
        } catch (Exception var11) {
            this.sendExceptionMessage("[takeSnapshot] Internal error was occurred. Unable to send data to gds.", var11);
            return false;
        }
    }

    private void saveBitmap(Bitmap saveImage) throws SecurityException, IOException, NullPointerException {
        Log.v("KiboRpcApi", "[Start] saveBitmap");
        String filepath = this.gsService.getGuestScienceDataBasePath() + "/immediate/JudgeImages";
        File file = new File(filepath);
        if (!file.exists()) {
            Log.v("KiboRpcApi", "[saveBitmap] Make save directory");
            file.mkdir();
        }

        Date mDate = new Date();
        SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String fileName = "snapshot_" + fileNameDate.format(mDate) + ".png";
        String AttachName = file.getAbsolutePath() + "/" + fileName;
        FileOutputStream out = new FileOutputStream(AttachName);
        saveImage.compress(CompressFormat.PNG, 100, out);
        out.flush();
        out.close();
        Log.v("KiboRpcApi", "[Finish] saveBitmap");
    }

    public double[][] getNavCamIntrinsics() {
        Log.i("KiboRpcApi", "[Start] getNavCamIntrinsics");
        double[][] camera_param = new double[2][];
        if (this.getterNode.getOnSimulation()) {
            Log.v("KiboRpcApi", "[getNavCamIntrinsics] getOnSimulation: True");
            Log.v("KiboRpcApi", "[getNavCamIntrinsics] Set simulation param");
            camera_param[0] = this.NAVCAM_CAMERA_MATRIX_SIMULATION;
            camera_param[1] = this.NAVCAM_DISTORTION_COEFFICIENTS_SIMULATION;
        } else {
            Log.v("KiboRpcApi", "[getNavCamIntrinsics] Set ISS param");
            camera_param[0] = this.NAVCAM_CAMERA_MATRIX_ISS;
            camera_param[1] = this.NAVCAM_DISTORTION_COEFFICIENTS_ISS;
        }

        Log.i("KiboRpcApi", "[Finish] getNavCamIntrinsics");
        return camera_param;
    }

    public double[][] getDockCamIntrinsics() {
        Log.i("KiboRpcApi", "[Start] getDockCamIntrinsics");
        double[][] camera_param = new double[2][];
        if (this.getterNode.getOnSimulation()) {
            Log.v("KiboRpcApi", "[getDockCamIntrinsics] getOnSimulation: True");
            Log.v("KiboRpcApi", "[getDockCamIntrinsics] Set simulation param");
            camera_param[0] = this.DOCKCAM_CAMERA_MATRIX_SIMULATION;
            camera_param[1] = this.DOCKCAM_DISTORTION_COEFFICIENTS_SIMULATION;
        } else {
            Log.v("KiboRpcApi", "[getDockCamIntrinsics] Set ISS param");
            camera_param[0] = this.DOCKCAM_CAMERA_MATRIX_ISS;
            camera_param[1] = this.DOCKCAM_DISTORTION_COEFFICIENTS_ISS;
        }

        Log.i("KiboRpcApi", "[Finish] getDockCamIntrinsics");
        return camera_param;
    }

    public void saveBitmapImage(Bitmap image, String imageName) {
        try {
            Log.i("KiboRpcApi", "[Start] saveBitmapImage");
            boolean checkArgs = true;
            if (image == null) {
                Log.e("KiboRpcApi", "[saveBitmapImage] image is null");
                checkArgs = false;
            }

            if (imageName == null) {
                Log.e("KiboRpcApi", "[saveBitmapImage] imageName is null");
                checkArgs = false;
            }

            if (checkArgs) {
                Log.i("KiboRpcApi", "Parameters: image, imageName == " + imageName);
                Log.v("KiboRpcApi", "[saveBitmapImage] Check directry");
                String filepath = this.gsService.getGuestScienceDataBasePath() + "/immediate/DebugImages";
                File file = new File(filepath);
                if (!file.exists()) {
                    Log.v("KiboRpcApi", "[saveBitmapImage] Make save directry");
                    file.mkdir();
                }

                File[] list = file.listFiles();
                int width = image.getWidth();
                int height = image.getHeight();
                int size = width * height;
                if (list.length >= 50) {
                    Log.e("KiboRpcApi", "[saveBitmapImage] Can't save more than 50 images");
                } else if (size > 1228800) {
                    Log.e("KiboRpcApi", "[saveBitmapImage] The size is too large.");
                } else {
                    Log.v("KiboRpcApi", "[saveBitmapImage] Save bitmap image");
                    this.saveImage(image, imageName, file);
                }
            }

            Log.i("KiboRpcApi", "[Finish] saveBitmapImage");
        } catch (Exception var10) {
            this.sendExceptionMessage("[saveBitmapImage] Internal error was occurred.", var10);
        }

    }

    public void saveMatImage(Mat image, String imageName) {
        try {
            Log.i("KiboRpcApi", "[Start] saveMatImage");
            boolean checkArgs = true;
            if (image == null) {
                Log.e("KiboRpcApi", "[saveMatImage] image is null");
                checkArgs = false;
            }

            if (imageName == null) {
                Log.e("KiboRpcApi", "[saveMatImage] imageName is null");
                checkArgs = false;
            }

            if (checkArgs) {
                Log.i("KiboRpcApi", "Parameters: image, imageName == " + imageName);
                Log.v("KiboRpcApi", "[saveMatImage] Check directry");
                String filepath = this.gsService.getGuestScienceDataBasePath() + "/immediate/DebugImages";
                File file = new File(filepath);
                if (!file.exists()) {
                    Log.v("KiboRpcApi", "[saveMatImage] make save directory");
                    file.mkdir();
                }

                File[] list = file.listFiles();
                int width = image.width();
                int height = image.height();
                int size = width * height;
                if (list.length >= 50) {
                    Log.e("KiboRpcApi", "[saveMatImage] Can't save more than 50 images.");
                } else if (size > 1228800) {
                    Log.e("KiboRpcApi", "[saveMatImage] The size is too large.");
                } else {
                    Log.v("KiboRpcApi", "[saveMatImage] Save mat image");
                    Bitmap bitmapImage = Bitmap.createBitmap(image.width(), image.height(), Config.ARGB_8888);
                    Utils.matToBitmap(image, bitmapImage);
                    this.saveImage(bitmapImage, imageName, file);
                }
            }

            Log.i("KiboRpcApi", "[Finish] saveMatImage");
        } catch (Exception var11) {
            this.sendExceptionMessage("[saveMatImage] Internal error was occurred.", var11);
        }

    }

    private void saveImage(Bitmap image, String imageName, File file) throws SecurityException, IOException, NullPointerException {
        String AttachName = file.getAbsolutePath() + "/" + imageName;
        Log.v("KiboRpcApi", AttachName);
        FileOutputStream out = new FileOutputStream(AttachName);
        image.compress(CompressFormat.PNG, 100, out);
        out.flush();
        out.close();
    }

    public void setAreaInfo(int areaId, String itemName) {
        this.setAreaInfo(areaId, itemName, 1);
    }

    public void setAreaInfo(int areaId, String itemName, int number) {
        try {
            Log.i("KiboRpcApi", "[Start] setAreaInfo");
            boolean checkArgs = true;
            if (itemName == null) {
                Log.e("KiboRpcApi", "[saveBitmapImage] itemName is null");
                itemName = "";
            }

            Log.i("KiboRpcApi", "Parameters: areaId == " + areaId + ", itemName == " + itemName + ", number == " + number);
            this.areaItemMap.setAreaInfo(areaId, itemName, number);
            JSONObject data = new JSONObject();
            data.put("area_id", areaId);
            data.put("lost_item", itemName);
            data.put("num", number);
            this.gsService.sendData(MessageType.JSON, "data", data.toString());
            Log.i("KiboRpcApi", "[Finish] setAreaInfo");
        } catch (Exception var6) {
            this.sendExceptionMessage("[setAreaInfo] Internal error was occurred.", var6);
        }

    }
}
```

## 3. Important Notes

The type of `api` is `KiboRpcApi`, you can import it as follows:

```java
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;
```