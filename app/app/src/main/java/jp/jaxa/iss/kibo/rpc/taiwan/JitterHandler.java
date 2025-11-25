package jp.jaxa.iss.kibo.rpc.taiwan;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;

import android.util.Log;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Class to handle gravity jitter events
 */
public class JitterHandler {
    private final String TAG = this.getClass().getSimpleName();
    private final KiboRpcApi api;
    private final Navigator navigator;
    private ScheduledExecutorService scheduler;
    private long monitorIntervalMs = 1000;

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

    public void start() {
        final Pose targetPose = navigator.getTargetPose();
        if (targetPose == null) {
            Log.w(TAG, "Jitter handler start failed: target pose is null.");
            return;
        }

        final Point targetPoint = targetPose.getPoint();
        final Quaternion targetQuat = targetPose.getQuaternion();

        // Define the monitoring task
        Runnable monitorTask = new Runnable() {
            @Override
            public void run() {
                if (Thread.currentThread().isInterrupted()) return;
                if (navigator.isMoving()) return;

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
                double angleRad = 2.0 * Math.acos(dot);
                double angleDeg = Math.toDegrees(angleRad);

                // Thresholds
                double MIN_DIST = 0.15; // 0.15 m
                double MIN_ANGLE = 20; // 20 degrees

                // Recover to the target pose
                if (dist > MIN_DIST || angleDeg > MIN_ANGLE) {
                    Log.w(TAG, String.format("Jitter detected! Dist: %.3f m, Angle: %.1f deg. Correcting...", dist, angleDeg));

                    // Only correct if not already moving (Navigator handles this check internally too, but good to check here)
                    if (!navigator.isMoving()) {
                        navigator.moveTo(targetPose);
                    }
                }
            }
        };

        if (scheduler == null || scheduler.isShutdown()) {
            scheduler = Executors.newScheduledThreadPool(1);
        }

        scheduler.scheduleAtFixedRate(
            monitorTask,
            0, // initial delay
            monitorIntervalMs,
            TimeUnit.MILLISECONDS
        );

        Log.i(TAG, "Jitter handler started.");
    }

    public void stop() {
        if (scheduler != null) {
            try {
                scheduler.shutdown();
                if (!scheduler.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
            }
            scheduler = null;
        }
        Log.i(TAG, "Jitter handler stopped.");
    }
}
