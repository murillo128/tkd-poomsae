package dev.murillo.tkd.multicam;

import android.content.Context;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.SystemClock;

import org.json.JSONObject;

import java.net.DatagramPacket;
import java.net.MulticastSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
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
    private static final String MULTICAST_GROUP = "239.255.42.99";
    private static final String PROTOCOL = "tkd-multicam-v2";
    private static final long PEER_TIMEOUT_NS = 8_000_000_000L;

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
            return bestRttNs == Long.MAX_VALUE ? Double.NaN : bestRttNs / 1_000_000.0;
        }
    }

    public interface Listener {
        void onPeersChanged();
        void onNetworkStatus(String status);

        // Commands received while acting as a camera node.
        void onArmCommand(String sessionId);
        void onStartCommand(long localTargetNs, long fallbackDelayMs);
        void onStopCommand();
    }

    private final Context context;
    private final Listener listener;
    private final String deviceId;
    private final String deviceName;
    private final Map<String, Peer> peers = new ConcurrentHashMap<>();

    private volatile Role role;
    private volatile boolean running;
    private volatile InetAddress controllerAddress;
    private MulticastSocket socket;
    private InetAddress multicastGroup;
    private WifiManager.MulticastLock multicastLock;
    private Thread receiveThread;
    private ScheduledExecutorService scheduler;

    public NetworkCoordinator(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        this.deviceId = context.getSharedPreferences("network", Context.MODE_PRIVATE)
                .getString("device_id", null) == null
                ? createAndPersistId(context)
                : context.getSharedPreferences("network", Context.MODE_PRIVATE)
                        .getString("device_id", UUID.randomUUID().toString());
        this.deviceName = Build.MANUFACTURER + " " + Build.MODEL;
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

        try {
            WifiManager wifi =
                    (WifiManager) context.getApplicationContext()
                            .getSystemService(Context.WIFI_SERVICE);
            multicastLock = wifi.createMulticastLock("tkd-multicam-discovery");
            multicastLock.setReferenceCounted(false);
            multicastLock.acquire();

            multicastGroup = InetAddress.getByName(MULTICAST_GROUP);
            socket = new MulticastSocket(null);
            socket.setReuseAddress(true);
            socket.bind(new InetSocketAddress(PORT));
            socket.setTimeToLive(1);
            socket.joinGroup(multicastGroup);
        } catch (Exception e) {
            running = false;
            releaseMulticastLock();
            listener.onNetworkStatus("Network start failed: " + describe(e));
            return;
        }

        receiveThread = new Thread(this::receiveLoop, "TkdMultiCamUdp");
        receiveThread.start();

        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "TkdMultiCamDiscovery");
            t.setDaemon(true);
            return t;
        });

        if (role == Role.CONTROLLER) {
            scheduler.scheduleAtFixedRate(() -> {
                if (!running) return;
                discoverNow();
                syncKnownPeers();
                prunePeers();
            }, 0, 1200, TimeUnit.MILLISECONDS);
            listener.onNetworkStatus("Controller discovery active on UDP " + PORT);
        } else {
            scheduler.scheduleAtFixedRate(() -> {
                if (!running) return;
                // Camera nodes also announce themselves periodically so a controller
                // appearing later does not need to wait for the next broadcast cycle.
                announceCamera();
            }, 0, 2000, TimeUnit.MILLISECONDS);
            listener.onNetworkStatus("Camera node waiting on UDP " + PORT);
        }
    }

    public synchronized void stop() {
        running = false;

        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }

        if (socket != null) {
            try {
                if (multicastGroup != null) socket.leaveGroup(multicastGroup);
            } catch (Exception ignored) {}
            socket.close();
            socket = null;
        }
        multicastGroup = null;
        releaseMulticastLock();

        if (receiveThread != null) {
            receiveThread.interrupt();
            receiveThread = null;
        }

        controllerAddress = null;
    }

    public List<Peer> getPeers() {
        long now = SystemClock.elapsedRealtimeNanos();
        List<Peer> list = new ArrayList<>();
        for (Peer peer : peers.values()) {
            if (now - peer.lastSeenNs <= PEER_TIMEOUT_NS) {
                list.add(peer);
            }
        }
        list.sort(Comparator.comparing(p -> p.name == null ? p.id : p.name));
        return list;
    }

    public void discoverNow() {
        if (!running || role != Role.CONTROLLER) return;
        try {
            JSONObject j = base("discover");
            sendBroadcast(j);
        } catch (Exception e) {
            listener.onNetworkStatus("Discover send failed: " + describe(e));
        }
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

    public void startAll(long targetControllerNs, long fallbackDelayMs) {
        if (role != Role.CONTROLLER) return;
        for (Peer peer : getPeers()) {
            try {
                JSONObject j = base("start");
                if (peer.hasSync) {
                    j.put("targetLocalNs", targetControllerNs + peer.clockOffsetNs);
                    j.put("fallbackDelayMs", fallbackDelayMs);
                } else {
                    j.put("targetLocalNs", 0L);
                    j.put("fallbackDelayMs", fallbackDelayMs);
                }
                sendReliableTo(peer.address, j);
                peer.status = peer.hasSync ? "start scheduled" : "start scheduled (delay fallback)";
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
                sendReliableTo(peer.address, base("stop"));
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
            listener.onNetworkStatus("READY send failed: " + describe(e));
        }
    }

    public void sendStarted(long localStartNs) {
        if (role != Role.CAMERA || controllerAddress == null) return;
        try {
            JSONObject j = base("started");
            j.put("localStartNs", localStartNs);
            sendTo(controllerAddress, j);
        } catch (Exception e) {
            listener.onNetworkStatus("STARTED send failed: " + describe(e));
        }
    }

    public void sendStopped(String path) {
        if (role != Role.CAMERA || controllerAddress == null) return;
        try {
            JSONObject j = base("stopped");
            if (path != null) j.put("path", path);
            sendTo(controllerAddress, j);
        } catch (Exception e) {
            listener.onNetworkStatus("STOPPED send failed: " + describe(e));
        }
    }

    public void sendCameraError(String error) {
        if (role != Role.CAMERA || controllerAddress == null) return;
        try {
            JSONObject j = base("camera_error");
            j.put("error", error);
            sendTo(controllerAddress, j);
        } catch (Exception ignored) {}
    }

    private void receiveLoop() {
        byte[] buffer = new byte[8192];

        while (running) {
            try {
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                socket.receive(packet);

                String raw = new String(
                        packet.getData(), packet.getOffset(), packet.getLength(),
                        StandardCharsets.UTF_8);
                JSONObject j = new JSONObject(raw);
                if (!PROTOCOL.equals(j.optString("protocol"))) {
                    continue;
                }

                String senderId = j.optString("senderId");
                if (deviceId.equals(senderId)) {
                    continue;
                }

                String type = j.optString("type");
                long receivedNs = SystemClock.elapsedRealtimeNanos();

                if (role == Role.CAMERA) {
                    handleAsCamera(type, j, packet.getAddress(), receivedNs);
                } else if (role == Role.CONTROLLER) {
                    handleAsController(type, j, packet.getAddress(), receivedNs);
                }
            } catch (Exception e) {
                if (running) {
                    listener.onNetworkStatus("UDP receive warning: " + describe(e));
                }
            }
        }
    }

    private void handleAsCamera(
            String type, JSONObject j, InetAddress senderAddress, long receivedNs) throws Exception {
        switch (type) {
            case "discover":
                controllerAddress = senderAddress;
                replyHello(senderAddress);
                break;

            case "controller_announce":
                controllerAddress = senderAddress;
                replyHello(senderAddress);
                break;

            case "sync": {
                controllerAddress = senderAddress;
                long t0 = j.getLong("t0");
                long t1 = receivedNs;
                long t2 = SystemClock.elapsedRealtimeNanos();
                JSONObject reply = base("sync_reply");
                reply.put("t0", t0);
                reply.put("t1", t1);
                reply.put("t2", t2);
                sendTo(senderAddress, reply);
                break;
            }

            case "arm":
                controllerAddress = senderAddress;
                listener.onArmCommand(j.optString(
                        "sessionId", "session-" + System.currentTimeMillis()));
                break;

            case "start": {
                controllerAddress = senderAddress;
                long target = j.optLong("targetLocalNs", 0L);
                long fallback = j.optLong("fallbackDelayMs", 3000L);
                listener.onStartCommand(target, fallback);
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
            String type, JSONObject j, InetAddress senderAddress, long receivedNs) throws Exception {
        String senderId = j.optString("senderId");
        if (senderId == null || senderId.isEmpty()) return;

        Peer peer = peers.computeIfAbsent(senderId, Peer::new);
        peer.name = j.optString("senderName", senderId);
        peer.address = senderAddress;
        peer.lastSeenNs = receivedNs;

        switch (type) {
            case "hello":
            case "camera_announce":
                if (!peer.ready && !"recording".equals(peer.status)) {
                    peer.status = "discovered";
                }
                break;

            case "sync_reply": {
                long t0 = j.getLong("t0");
                long t1 = j.getLong("t1");
                long t2 = j.getLong("t2");
                long t3 = receivedNs;

                long rtt = (t3 - t0) - (t2 - t1);
                long offset = ((t1 - t0) + (t2 - t3)) / 2L;

                if (rtt >= 0 && rtt < peer.bestRttNs) {
                    peer.bestRttNs = rtt;
                    peer.clockOffsetNs = offset;
                    peer.hasSync = true;
                }
                break;
            }

            case "ready":
                peer.ready = true;
                peer.status = j.optString("detail", "ready");
                break;

            case "started":
                peer.status = "recording";
                break;

            case "stopped":
                peer.ready = false;
                peer.status = "stopped";
                break;

            case "camera_error":
                peer.ready = false;
                peer.status = "ERROR: " + j.optString("error", "unknown");
                break;

            default:
                break;
        }

        listener.onPeersChanged();
    }

    private void replyHello(InetAddress address) throws Exception {
        JSONObject j = base("hello");
        j.put("role", "camera");
        sendTo(address, j);
    }

    private void announceCamera() {
        if (!running || role != Role.CAMERA) return;
        try {
            sendBroadcast(base("camera_announce"));
        } catch (Exception ignored) {}
    }

    private void syncKnownPeers() {
        if (!running || role != Role.CONTROLLER) return;

        // Announce the controller as well, which helps on Wi-Fi networks where
        // directed broadcast behavior is inconsistent.
        try {
            sendBroadcast(base("controller_announce"));
        } catch (Exception ignored) {}

        for (Peer peer : getPeers()) {
            if (peer.address == null) continue;
            try {
                JSONObject j = base("sync");
                j.put("t0", SystemClock.elapsedRealtimeNanos());
                sendTo(peer.address, j);
            } catch (Exception ignored) {}
        }
    }

    private void prunePeers() {
        long now = SystemClock.elapsedRealtimeNanos();
        boolean changed = false;
        for (Map.Entry<String, Peer> entry : peers.entrySet()) {
            if (now - entry.getValue().lastSeenNs > PEER_TIMEOUT_NS) {
                peers.remove(entry.getKey());
                changed = true;
            }
        }
        if (changed) listener.onPeersChanged();
    }

    private JSONObject base(String type) throws Exception {
        JSONObject j = new JSONObject();
        j.put("protocol", PROTOCOL);
        j.put("type", type);
        j.put("senderId", deviceId);
        j.put("senderName", deviceName);
        return j;
    }

    private void sendBroadcast(JSONObject j) throws Exception {
        MulticastSocket current = socket;
        InetAddress group = multicastGroup;
        if (current == null || current.isClosed() || group == null) {
            throw new IllegalStateException("Multicast socket is not active");
        }

        byte[] data = j.toString().getBytes(StandardCharsets.UTF_8);
        DatagramPacket packet = new DatagramPacket(data, data.length, group, PORT);
        current.send(packet);
    }

    private void sendTo(InetAddress address, JSONObject j) throws Exception {
        sendRaw(address, j.toString());
    }

    private void sendReliableTo(InetAddress address, JSONObject j) throws Exception {
        final String payload = j.toString();
        sendRaw(address, payload);

        ScheduledExecutorService s = scheduler;
        if (s != null && !s.isShutdown()) {
            s.schedule(() -> {
                try { sendRaw(address, payload); } catch (Exception ignored) {}
            }, 40, TimeUnit.MILLISECONDS);
            s.schedule(() -> {
                try { sendRaw(address, payload); } catch (Exception ignored) {}
            }, 100, TimeUnit.MILLISECONDS);
        }
    }

    private void sendRaw(InetAddress address, String payload) throws Exception {
        MulticastSocket current = socket;
        if (address == null || current == null || current.isClosed()) return;
        byte[] data = payload.getBytes(StandardCharsets.UTF_8);
        DatagramPacket packet = new DatagramPacket(data, data.length, address, PORT);
        current.send(packet);
    }

    private void releaseMulticastLock() {
        try {
            if (multicastLock != null && multicastLock.isHeld()) {
                multicastLock.release();
            }
        } catch (Exception ignored) {
        }
        multicastLock = null;
    }

    private static String describe(Exception e) {
        String msg = e.getMessage();
        return e.getClass().getSimpleName() + (msg == null ? "" : ": " + msg);
    }
}
