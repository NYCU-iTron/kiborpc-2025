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

    private long monitorIntervalMs = 1000; // milliseconds
    private final double minDist = 0.10; // meters
    private final double minAngle = 20; // degrees

    private volatile boolean isRunning = false;

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
        // Get the target pose to monitor
        final Pose targetPose = navigator.getTargetPose();
        if (targetPose == null) {
            Log.w(TAG, "Jitter handler start failed: target pose is null.");
            return;
        }

        final Point targetPoint = targetPose.getPoint();
        final Quaternion targetQuat = targetPose.getQuaternion();

        isRunning = true;

        // Define the monitoring task
        Runnable monitorTask = new Runnable() {
            @Override
            public void run() {
                if (!isRunning) return;
                if (navigator.isMoving()) return;
                if (Thread.currentThread().isInterrupted()) return;

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

                    if (!isRunning) return;
                    if (navigator.isMoving()) return;
                    if (Thread.currentThread().isInterrupted()) return;

                    navigator.moveTo(targetPose);
                }
            }
        };

        // Initialize the scheduler
        if (scheduler == null || scheduler.isShutdown()) {
            scheduler = Executors.newSingleThreadScheduledExecutor();
        }

        // Schedule the monitoring task at fixed delay after previous task completion
        scheduler.scheduleWithFixedDelay(
            monitorTask,
            0, // initial delay
            monitorIntervalMs,
            TimeUnit.MILLISECONDS
        );

        Log.i(TAG, "Jitter handler started.");
    }

    public void stop() {
        isRunning = false;
        if (scheduler == null || scheduler.isShutdown()) {
            Log.i(TAG, "Jitter handler is not running.");
            return;
        }

        try {
            scheduler.shutdown();

            // Wait for existing tasks to terminate
            if (!scheduler.awaitTermination(30, TimeUnit.SECONDS)) {
                Log.w(TAG, "Jitter handler did not terminate in time, forcing shutdown.");
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted while waiting for JitterHandler to stop.");
            scheduler.shutdownNow();
        }

        // Clear the scheduler reference
        scheduler = null;

        Log.i(TAG, "Jitter handler stopped.");
    }
}
