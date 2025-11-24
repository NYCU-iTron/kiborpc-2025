package jp.jaxa.iss.kibo.rpc.sampleapk;

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
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    private double currentAcceleration = 0.0;
    private double accelerationThreshold = 0.05; // m/s^2
    private final long monitorIntervalMs = 100; // ms

    enum State {
        IDLE,
        MONITORING,
        RECOVERING,
    }
    private State state = State.IDLE;
    private boolean isJitterDetected = false;

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

    private final Runnable monitorTask = () -> {
        currentAcceleration = calculateAcceleration();
        if (currentAcceleration > accelerationThreshold) {
            isJitterDetected = true;
        }

        try {
            if (isJitterDetected && state != State.RECOVERING) {
                Log.w(TAG, "Jitter detected! Start recovery...");
                state = State.RECOVERING;
                // navigator.recoverFromJitter();
            } else if (isJitterDetected && state == State.RECOVERING) {
                Log.i(TAG, "Jitter detected! Still recovering...");
                // Still in jitter state
                isJitterDetected = false; // Reset for next check
            } else if (state == State.RECOVERING) {
                Log.i(TAG, "Jitter stopped. Resume normal task.");
                state = State.MONITORING;
                // navigator.resumeNormalTask();

            }
        } catch (Exception e) {
            Log.e(TAG, "Monitor error: " + e.getMessage());
        }
    };

    public void startMonitoring() {
        scheduler.scheduleAtFixedRate(
            monitorTask,
            0, // initial delay
            monitorIntervalMs,
            TimeUnit.MILLISECONDS
        );
        state = State.MONITORING;
        Log.i(TAG, "Jitter monitor started.");
    }

    public void stopMonitoring() {
        try {
            scheduler.shutdown();
            if (!scheduler.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
        }
        state = State.IDLE;
        Log.i(TAG, "Jitter monitor stopped.");
    }

    private double calculateAcceleration() {
        Kinematics kinematics = api.getKinematics();
        if (kinematics == null) {
            Log.e(TAG, "Failed to get kinematics.");
            return 0.0;
        }

        Vec3d linearAccel = kinematics.getLinearAcceleration();
        if (linearAccel == null) {
            Log.e(TAG, "Failed to get linear acceleration.");
            return 0.0;
        }

        return Math.sqrt(
            linearAccel.x * linearAccel.x +
            linearAccel.y * linearAccel.y +
            linearAccel.z * linearAccel.z
        );
    }
}
