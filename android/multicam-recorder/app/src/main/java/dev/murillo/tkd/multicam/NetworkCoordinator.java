package dev.murillo.tkd.multicam;

import android.content.Context;
import android.net.nsd.NsdManager;
import android.net.nsd.NsdServiceInfo;
import android.os.Build;
import android.os.SystemClock;

import org.json.JSONObject;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public final class NetworkCoordinator {
    public static final int PORT = 45873;
    private static final String SERVICE_TYPE = "_tkdmulticam._udp.";
    private static final String PROTOCOL = "tkd-multicam-v3";
    private static final long PEER_TIMEOUT_NS = 12_000_000_000L;

    public enum Role {
        CONTROLLER, CAMERA
    }

    public static final class Peer {
        public final String id;
        public volatile String name;
        public volatile InetAddress address;
        public volatile long lastSeenNs;
        public volatile boolean ready;
        public volatile String status = "discovered";
        public volatile boolean hasSync;
        public volatile long clockOffsetNs;
        public volatile long bestRttNs = Long.MAX_VALUE;

        Peer(String id) {
            this.id = id;
        }

        public double rttMs() {
            return bestRttNs == Long.MAX_VALUE
                    ? Double.NaN
                    : bestRttNs / 1_000_000.0;
        }
    }

    public interface Listener {
        void onPeersChanged();
        void onNetworkStatus(String status);
        void onArmCommand(String sessionId);
        void onStartCommand(long localTargetNs, long fallbackDelayMs);
        void onStopCommand();
    }

    private final Context context;
    private final Listener listener;
    private final String deviceId;
    private final String deviceName;
    private final Map<String, Peer> peers = new ConcurrentHashMap<>();

    private final NsdManager nsdManager;

    private volatile Role role;
    private volatile boolean running;
    private volatile InetAddress controllerAddress;

    private DatagramSocket socket;
    private Thread receiveThread;
    private ScheduledExecutorService scheduler;

    private NsdManager.DiscoveryListener discoveryListener;
    private NsdManager.RegistrationListener registrationListener;
    private volatile boolean serviceRegistered;
    private volatile boolean discoveryRunning;

    public NetworkCoordinator(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;

        String existing =
                context.getSharedPreferences("network", Context.MODE_PRIVATE)
                        .getString("device_id", null);

        this.deviceId =
                existing == null
                        ? createAndPersistId(context)
                        : existing;

        this.deviceName =
                Build.MANUFACTURER + " " + Build.MODEL;

        this.nsdManager =
                (NsdManager) this.context.getSystemService(Context.NSD_SERVICE);
    }

    private static String createAndPersistId(Context context) {
        String id = UUID.randomUUID().toString();

        context.getSharedPreferences("network", Context.MODE_PRIVATE)
                .edit()
                .putString("device_id", id)
                .apply();

        return id;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public String getDeviceName() {
        return deviceName;
    }

    public synchronized void start(Role newRole) {
        stop();

        role = newRole;
        running = true;
        peers.clear();
        controllerAddress = null;

        try {
            socket = new DatagramSocket(PORT);
            socket.setReuseAddress(true);
        } catch (Exception e) {
            running = false;
            listener.onNetworkStatus(
                    "Network start failed: " + describe(e));
            return;
        }

        receiveThread =
                new Thread(this::receiveLoop, "TkdMultiCamUdp");
        receiveThread.start();

        scheduler =
                Executors.newSingleThreadScheduledExecutor(r -> {
                    Thread t =
                            new Thread(r, "TkdMultiCamSync");
                    t.setDaemon(true);
                    return t;
                });

        if (role == Role.CONTROLLER) {
            startDiscovery();

            scheduler.scheduleAtFixedRate(() -> {
                if (!running) return;
                syncKnownPeers();
                prunePeers();
            }, 300, 900, TimeUnit.MILLISECONDS);

            listener.onNetworkStatus(
                    "Auto-discovery active · NSD/mDNS");
        } else {
            registerCameraService();

            listener.onNetworkStatus(
                    "Camera advertised on local Wi‑Fi");
        }
    }

    public synchronized void stop() {
        running = false;

        stopDiscovery();
        unregisterCameraService();

        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }

        if (socket != null) {
            socket.close();
            socket = null;
        }

        if (receiveThread != null) {
            receiveThread.interrupt();
            receiveThread = null;
        }

        controllerAddress = null;
    }

    public List<Peer> getPeers() {
        long now =
                SystemClock.elapsedRealtimeNanos();

        List<Peer> list =
                new ArrayList<>();

        for (Peer peer : peers.values()) {
            if (now - peer.lastSeenNs <= PEER_TIMEOUT_NS) {
                list.add(peer);
            }
        }

        list.sort(
                Comparator.comparing(
                        p -> p.name == null ? p.id : p.name));

        return list;
    }

    public void discoverNow() {
        if (!running || role != Role.CONTROLLER) return;

        // NSD continuously discovers services. Restarting it gives the UI button
        // a deterministic refresh action without using raw multicast packets.
        stopDiscovery();
        startDiscovery();
    }

    public void armAll(String sessionId) {
        if (role != Role.CONTROLLER) return;

        for (Peer peer : getPeers()) {
            peer.ready = false;
            peer.status = "arming";

            try {
                JSONObject j = base("arm");
                j.put("sessionId", sessionId);
                sendReliableTo(peer.address, j);
            } catch (Exception e) {
                peer.status = "arm send failed";
            }
        }

        listener.onPeersChanged();
    }

    public void startAll(
            long targetControllerNs,
            long fallbackDelayMs) {

        if (role != Role.CONTROLLER) return;

        for (Peer peer : getPeers()) {
            try {
                JSONObject j = base("start");

                if (peer.hasSync) {
                    j.put(
                            "targetLocalNs",
                            targetControllerNs + peer.clockOffsetNs);
                } else {
                    j.put("targetLocalNs", 0L);
                }

                j.put("fallbackDelayMs", fallbackDelayMs);

                sendReliableTo(peer.address, j);

                peer.status =
                        peer.hasSync
                                ? "start scheduled"
                                : "start scheduled · fallback";
            } catch (Exception e) {
                peer.status = "start send failed";
            }
        }

        listener.onPeersChanged();
    }

    public void stopAll() {
        if (role != Role.CONTROLLER) return;

        for (Peer peer : getPeers()) {
            try {
                sendReliableTo(
                        peer.address,
                        base("stop"));

                peer.status = "stopping";
            } catch (Exception e) {
                peer.status = "stop send failed";
            }
        }

        listener.onPeersChanged();
    }

    public void sendReady(String detail) {
        if (role != Role.CAMERA || controllerAddress == null) return;

        try {
            JSONObject j = base("ready");
            j.put("detail", detail);
            sendTo(controllerAddress, j);
        } catch (Exception e) {
            listener.onNetworkStatus(
                    "READY send failed: " + describe(e));
        }
    }

    public void sendStarted(long localStartNs) {
        if (role != Role.CAMERA || controllerAddress == null) return;

        try {
            JSONObject j = base("started");
            j.put("localStartNs", localStartNs);
            sendTo(controllerAddress, j);
        } catch (Exception e) {
            listener.onNetworkStatus(
                    "STARTED send failed: " + describe(e));
        }
    }

    public void sendStopped(String path) {
        if (role != Role.CAMERA || controllerAddress == null) return;

        try {
            JSONObject j = base("stopped");
            if (path != null) j.put("path", path);
            sendTo(controllerAddress, j);
        } catch (Exception e) {
            listener.onNetworkStatus(
                    "STOPPED send failed: " + describe(e));
        }
    }

    public void sendCameraError(String error) {
        if (role != Role.CAMERA || controllerAddress == null) return;

        try {
            JSONObject j = base("camera_error");
            j.put("error", error);
            sendTo(controllerAddress, j);
        } catch (Exception ignored) {
        }
    }

    private void startDiscovery() {
        if (!running
                || role != Role.CONTROLLER
                || discoveryRunning
                || nsdManager == null) {
            return;
        }

        discoveryListener =
                new NsdManager.DiscoveryListener() {
                    @Override
                    public void onDiscoveryStarted(
                            String serviceType) {
                        discoveryRunning = true;
                        listener.onNetworkStatus(
                                "Searching local Wi‑Fi · NSD");
                    }

                    @Override
                    public void onServiceFound(
                            NsdServiceInfo serviceInfo) {

                        if (!running
                                || role != Role.CONTROLLER) {
                            return;
                        }

                        if (!SERVICE_TYPE.equals(
                                serviceInfo.getServiceType())) {
                            return;
                        }

                        resolveService(serviceInfo);
                    }

                    @Override
                    public void onServiceLost(
                            NsdServiceInfo serviceInfo) {
                        String serviceName =
                                serviceInfo.getServiceName();

                        boolean changed = false;

                        for (Map.Entry<String, Peer> entry
                                : peers.entrySet()) {

                            Peer peer = entry.getValue();

                            if (serviceName.equals(peer.name)) {
                                peers.remove(entry.getKey());
                                changed = true;
                            }
                        }

                        if (changed) {
                            listener.onPeersChanged();
                        }
                    }

                    @Override
                    public void onDiscoveryStopped(
                            String serviceType) {
                        discoveryRunning = false;
                    }

                    @Override
                    public void onStartDiscoveryFailed(
                            String serviceType,
                            int errorCode) {
                        discoveryRunning = false;

                        listener.onNetworkStatus(
                                "NSD discovery failed · " + errorCode);
                    }

                    @Override
                    public void onStopDiscoveryFailed(
                            String serviceType,
                            int errorCode) {
                        discoveryRunning = false;
                    }
                };

        try {
            nsdManager.discoverServices(
                    SERVICE_TYPE,
                    NsdManager.PROTOCOL_DNS_SD,
                    discoveryListener);
        } catch (Exception e) {
            discoveryRunning = false;

            listener.onNetworkStatus(
                    "NSD discovery failed: " + describe(e));
        }
    }

    private void stopDiscovery() {
        if (nsdManager == null
                || discoveryListener == null
                || !discoveryRunning) {
            discoveryListener = null;
            discoveryRunning = false;
            return;
        }

        try {
            nsdManager.stopServiceDiscovery(
                    discoveryListener);
        } catch (Exception ignored) {
        }

        discoveryListener = null;
        discoveryRunning = false;
    }

    private void resolveService(
            NsdServiceInfo serviceInfo) {

        try {
            nsdManager.resolveService(
                    serviceInfo,
                    new NsdManager.ResolveListener() {
                        @Override
                        public void onResolveFailed(
                                NsdServiceInfo serviceInfo,
                                int errorCode) {
                        }

                        @Override
                        public void onServiceResolved(
                                NsdServiceInfo resolved) {

                            if (!running
                                    || role != Role.CONTROLLER) {
                                return;
                            }

                            InetAddress address =
                                    resolved.getHost();

                            if (address == null) return;

                            String resolvedId = null;

                            try {
                                byte[] idBytes =
                                        resolved.getAttributes().get("id");

                                if (idBytes != null) {
                                    resolvedId =
                                            new String(
                                                    idBytes,
                                                    StandardCharsets.UTF_8);
                                }
                            } catch (Exception ignored) {
                            }

                            if (resolvedId == null
                                    || resolvedId.isEmpty()) {
                                resolvedId =
                                        resolved.getServiceName();
                            }

                            if (deviceId.equals(resolvedId)) return;

                            Peer peer =
                                    peers.computeIfAbsent(
                                            resolvedId,
                                            Peer::new);

                            peer.name =
                                    resolved.getServiceName();

                            peer.address = address;

                            peer.lastSeenNs =
                                    SystemClock.elapsedRealtimeNanos();

                            if (!peer.ready
                                    && !"recording".equals(peer.status)) {
                                peer.status = "discovered";
                            }

                            listener.onPeersChanged();

                            // Immediately establish the unicast path and clock sync.
                            sendSync(peer);
                        }
                    });
        } catch (Exception e) {
            listener.onNetworkStatus(
                    "NSD resolve warning: " + describe(e));
        }
    }

    private void registerCameraService() {
        if (!running
                || role != Role.CAMERA
                || nsdManager == null
                || serviceRegistered) {
            return;
        }

        NsdServiceInfo serviceInfo =
                new NsdServiceInfo();

        serviceInfo.setServiceName(
                "TKD-" + Build.MODEL + "-"
                        + deviceId.substring(0, 5));

        serviceInfo.setServiceType(
                SERVICE_TYPE);

        serviceInfo.setPort(PORT);

        try {
            serviceInfo.setAttribute(
                    "id",
                    deviceId);
        } catch (Exception ignored) {
        }

        registrationListener =
                new NsdManager.RegistrationListener() {
                    @Override
                    public void onServiceRegistered(
                            NsdServiceInfo serviceInfo) {
                        serviceRegistered = true;

                        listener.onNetworkStatus(
                                "Camera visible · "
                                        + serviceInfo.getServiceName());
                    }

                    @Override
                    public void onRegistrationFailed(
                            NsdServiceInfo serviceInfo,
                            int errorCode) {
                        serviceRegistered = false;

                        listener.onNetworkStatus(
                                "NSD register failed · " + errorCode);
                    }

                    @Override
                    public void onServiceUnregistered(
                            NsdServiceInfo serviceInfo) {
                        serviceRegistered = false;
                    }

                    @Override
                    public void onUnregistrationFailed(
                            NsdServiceInfo serviceInfo,
                            int errorCode) {
                        serviceRegistered = false;
                    }
                };

        try {
            nsdManager.registerService(
                    serviceInfo,
                    NsdManager.PROTOCOL_DNS_SD,
                    registrationListener);
        } catch (Exception e) {
            serviceRegistered = false;

            listener.onNetworkStatus(
                    "NSD register failed: " + describe(e));
        }
    }

    private void unregisterCameraService() {
        if (nsdManager == null
                || registrationListener == null
                || !serviceRegistered) {

            registrationListener = null;
            serviceRegistered = false;
            return;
        }

        try {
            nsdManager.unregisterService(
                    registrationListener);
        } catch (Exception ignored) {
        }

        registrationListener = null;
        serviceRegistered = false;
    }

    private void receiveLoop() {
        byte[] buffer = new byte[8192];

        while (running) {
            try {
                DatagramPacket packet =
                        new DatagramPacket(
                                buffer,
                                buffer.length);

                socket.receive(packet);

                String raw =
                        new String(
                                packet.getData(),
                                packet.getOffset(),
                                packet.getLength(),
                                StandardCharsets.UTF_8);

                JSONObject j =
                        new JSONObject(raw);

                if (!PROTOCOL.equals(
                        j.optString("protocol"))) {
                    continue;
                }

                String senderId =
                        j.optString("senderId");

                if (deviceId.equals(senderId)) {
                    continue;
                }

                String type =
                        j.optString("type");

                long receivedNs =
                        SystemClock.elapsedRealtimeNanos();

                if (role == Role.CAMERA) {
                    handleAsCamera(
                            type,
                            j,
                            packet.getAddress(),
                            receivedNs);
                } else if (role == Role.CONTROLLER) {
                    handleAsController(
                            type,
                            j,
                            packet.getAddress(),
                            receivedNs);
                }
            } catch (Exception e) {
                if (running) {
                    listener.onNetworkStatus(
                            "UDP warning: " + describe(e));
                }
            }
        }
    }

    private void handleAsCamera(
            String type,
            JSONObject j,
            InetAddress senderAddress,
            long receivedNs) throws Exception {

        switch (type) {
            case "sync": {
                controllerAddress = senderAddress;

                long t0 = j.getLong("t0");
                long t1 = receivedNs;
                long t2 =
                        SystemClock.elapsedRealtimeNanos();

                JSONObject reply =
                        base("sync_reply");

                reply.put("t0", t0);
                reply.put("t1", t1);
                reply.put("t2", t2);

                sendTo(senderAddress, reply);
                break;
            }

            case "arm":
                controllerAddress = senderAddress;

                listener.onArmCommand(
                        j.optString(
                                "sessionId",
                                "session-"
                                        + System.currentTimeMillis()));
                break;

            case "start": {
                controllerAddress = senderAddress;

                long target =
                        j.optLong(
                                "targetLocalNs",
                                0L);

                long fallback =
                        j.optLong(
                                "fallbackDelayMs",
                                3000L);

                listener.onStartCommand(
                        target,
                        fallback);
                break;
            }

            case "stop":
                controllerAddress = senderAddress;
                listener.onStopCommand();
                break;

            default:
                break;
        }
    }

    private void handleAsController(
            String type,
            JSONObject j,
            InetAddress senderAddress,
            long receivedNs) throws Exception {

        String senderId =
                j.optString("senderId");

        if (senderId == null
                || senderId.isEmpty()) {
            return;
        }

        Peer peer =
                peers.computeIfAbsent(
                        senderId,
                        Peer::new);

        peer.name =
                j.optString(
                        "senderName",
                        peer.name == null
                                ? senderId
                                : peer.name);

        peer.address = senderAddress;
        peer.lastSeenNs = receivedNs;

        switch (type) {
            case "sync_reply": {
                long t0 = j.getLong("t0");
                long t1 = j.getLong("t1");
                long t2 = j.getLong("t2");
                long t3 = receivedNs;

                long rtt =
                        (t3 - t0)
                                - (t2 - t1);

                long offset =
                        ((t1 - t0)
                                + (t2 - t3))
                                / 2L;

                if (rtt >= 0
                        && rtt < peer.bestRttNs) {

                    peer.bestRttNs = rtt;
                    peer.clockOffsetNs = offset;
                    peer.hasSync = true;
                }

                if (!peer.ready
                        && !"recording".equals(peer.status)) {
                    peer.status = "synced";
                }

                break;
            }

            case "ready":
                peer.ready = true;
                peer.status =
                        j.optString(
                                "detail",
                                "ready");
                break;

            case "started":
                peer.status = "recording";
                break;

            case "stopped":
                peer.ready = false;
                peer.status = "saved";
                break;

            case "camera_error":
                peer.ready = false;
                peer.status =
                        "ERROR · "
                                + j.optString(
                                        "error",
                                        "unknown");
                break;

            default:
                break;
        }

        listener.onPeersChanged();
    }

    private void syncKnownPeers() {
        if (!running
                || role != Role.CONTROLLER) {
            return;
        }

        for (Peer peer : getPeers()) {
            sendSync(peer);
        }
    }

    private void sendSync(Peer peer) {
        if (peer == null
                || peer.address == null) {
            return;
        }

        try {
            JSONObject j =
                    base("sync");

            j.put(
                    "t0",
                    SystemClock.elapsedRealtimeNanos());

            sendTo(peer.address, j);
        } catch (Exception ignored) {
        }
    }

    private void prunePeers() {
        long now =
                SystemClock.elapsedRealtimeNanos();

        boolean changed = false;

        for (Map.Entry<String, Peer> entry
                : peers.entrySet()) {

            if (now - entry.getValue().lastSeenNs
                    > PEER_TIMEOUT_NS) {

                peers.remove(entry.getKey());
                changed = true;
            }
        }

        if (changed) {
            listener.onPeersChanged();
        }
    }

    private JSONObject base(String type)
            throws Exception {

        JSONObject j =
                new JSONObject();

        j.put("protocol", PROTOCOL);
        j.put("type", type);
        j.put("senderId", deviceId);
        j.put("senderName", deviceName);

        return j;
    }

    private void sendTo(
            InetAddress address,
            JSONObject j) throws Exception {

        sendRaw(
                address,
                j.toString());
    }

    private void sendReliableTo(
            InetAddress address,
            JSONObject j) throws Exception {

        final String payload =
                j.toString();

        sendRaw(address, payload);

        ScheduledExecutorService s =
                scheduler;

        if (s != null
                && !s.isShutdown()) {

            s.schedule(() -> {
                try {
                    sendRaw(address, payload);
                } catch (Exception ignored) {}
            }, 40, TimeUnit.MILLISECONDS);

            s.schedule(() -> {
                try {
                    sendRaw(address, payload);
                } catch (Exception ignored) {}
            }, 100, TimeUnit.MILLISECONDS);
        }
    }

    private void sendRaw(
            InetAddress address,
            String payload) throws Exception {

        DatagramSocket current = socket;

        if (address == null
                || current == null
                || current.isClosed()) {
            return;
        }

        byte[] data =
                payload.getBytes(
                        StandardCharsets.UTF_8);

        DatagramPacket packet =
                new DatagramPacket(
                        data,
                        data.length,
                        address,
                        PORT);

        current.send(packet);
    }

    private static String describe(Exception e) {
        String msg = e.getMessage();

        return e.getClass().getSimpleName()
                + (msg == null ? "" : ": " + msg);
    }
}
