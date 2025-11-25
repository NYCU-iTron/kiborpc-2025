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
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
    private long monitorIntervalMs = 200;

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

    public void start(final int areaId) {
        Runnable monitorTask = new Runnable() {
            @Override
            public void run() {
                navigator.navigateToArea(areaId);
            }
        };

        scheduler.scheduleAtFixedRate(
            monitorTask,
            0, // initial delay
            monitorIntervalMs,
            TimeUnit.MILLISECONDS
        );

        Log.i(TAG, "Jitter handler started.");
    }

    public void stop() {
        try {
            scheduler.shutdown();
            if (!scheduler.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
        }

        Log.i(TAG, "Jitter handler stopped.");
    }
}
