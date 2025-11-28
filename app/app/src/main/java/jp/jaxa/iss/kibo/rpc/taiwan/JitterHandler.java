package jp.jaxa.iss.kibo.rpc.taiwan;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;

import android.util.Log;


/**
 * Class to handle gravity jitter events
 */
public class JitterHandler {
    private final String TAG = this.getClass().getSimpleName();
    private final KiboRpcApi api;
    private final Navigator navigator;

    private long monitorIntervalMs = 1000; // milliseconds
    private final double minDist = 0.10; // meters
    private final double minAngle = 20; // degrees

    /**
     * Constructor
     *
     * @param api KiboRpcApi instance
     * @param navigator Navigator instance
     */
    public JitterHandler(KiboRpcApi api, Navigator navigator) {
        this.api = api;
        this.navigator = navigator;

        Log.i(TAG, "Initialized.");
    }

    public void checkAndRecover() {
        // Get the target pose to monitor
        Pose targetPose = navigator.getTargetPose();
        if (targetPose == null) {
            Log.w(TAG, "Jitter handler start failed: target pose is null.");
            return;
        }

        Point targetPoint = targetPose.getPoint();
        Quaternion targetQuat = targetPose.getQuaternion();

        // Get current robot kinematics
        Kinematics kinematics = api.getRobotKinematics();
        Point currentPoint = kinematics.getPosition();
        Quaternion currentQuat = kinematics.getOrientation();

        // Calculate distance deviation
        double dist = Math.sqrt(
            Math.pow(currentPoint.getX() - targetPoint.getX(), 2) +
            Math.pow(currentPoint.getY() - targetPoint.getY(), 2) +
            Math.pow(currentPoint.getZ() - targetPoint.getZ(), 2)
        );

        // Calculate angle deviation (using quaternion dot product)
        // Angle = 2 * acos(|q1 . q2|)
        double dot = Math.abs(
            currentQuat.getX() * targetQuat.getX() +
            currentQuat.getY() * targetQuat.getY() +
            currentQuat.getZ() * targetQuat.getZ() +
            currentQuat.getW() * targetQuat.getW()
        );

        // Clamp dot product to [-1, 1] to avoid NaN
        if (dot > 1.0) dot = 1.0;
        double angle = 2.0 * Math.acos(dot);
        angle = Math.toDegrees(angle);

        // Recover to the target pose
        if (dist > minDist || angle > minAngle) {
            Log.w(TAG, String.format("Jitter detected! Dist: %.3f m, Angle: %.1f deg. Correcting...", dist, angle));
            navigator.moveTo(targetPose);
        }
    }
}
