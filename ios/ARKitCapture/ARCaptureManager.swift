import ARKit
import Observation

@Observable
final class ARCaptureManager: NSObject {

    // MARK: – Published state (read by SwiftUI)
    var isConnected  = false
    var statusText   = "Disconnected"
    var sentFPS: Double = 0
    var trackingState: String = "–"

    // MARK: – Settings (set before calling connect)
    var downsampling: Float  = 1.5
    var jpegQuality: CGFloat = 0.5
    /// Maximum frames per second sent to the server.
    var maxSendFPS: Double   = 15

    // MARK: – Private
    let arSession = ARSession()
    private var wsClient: WebSocketClient?
    private var encoder: FrameEncoder!

    private var lastSentAt: TimeInterval = 0
    private var sentCount  = 0
    private var fpsWindowStart: TimeInterval = 0

    private let delegateQueue = DispatchQueue(
        label: "arkit.delegate", qos: .userInteractive)

    override init() {
        super.init()
        arSession.delegateQueue = delegateQueue
        arSession.delegate = self
    }

    // MARK: – ARKit session

    func startARSession() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity          // Y up
        config.frameSemantics = []
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopARSession() {
        arSession.pause()
    }

    // MARK: – WebSocket connection

    func connect(host: String, port: Int) {
        guard let url = URL(string: "ws://\(host):\(port)") else {
            statusText = "Invalid address"
            return
        }
        encoder = FrameEncoder(downsampling: downsampling, jpegQuality: jpegQuality)

        let client = WebSocketClient(url: url)
        wsClient = client

        Task {
            await client.connect()
            let connected = await client.isConnected
            await MainActor.run {
                self.isConnected = connected
                self.statusText  = connected
                    ? "Connected → \(host):\(port)"
                    : "Connection failed"
            }
        }
    }

    func disconnect() {
        Task {
            await wsClient?.disconnect()
            wsClient = nil
            await MainActor.run {
                self.isConnected = false
                self.statusText  = "Disconnected"
            }
        }
    }
}

// MARK: – ARSessionDelegate
extension ARCaptureManager: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // ── Tracking state label ──────────────────────────────────────────────
        let stateStr: String
        switch frame.camera.trackingState {
        case .normal:                          stateStr = "Normal"
        case .limited(.initializing):         stateStr = "Initializing"
        case .limited(.relocalizing):         stateStr = "Relocalizing"
        case .limited(.insufficientFeatures): stateStr = "Low features"
        case .limited(.excessiveMotion):      stateStr = "Motion blur"
        case .notAvailable:                   stateStr = "Unavailable"
        @unknown default:                     stateStr = "Unknown"
        }
        DispatchQueue.main.async { self.trackingState = stateStr }

        // Only send when tracking is reliable
        guard case .normal = frame.camera.trackingState else { return }

        // ── Rate-limit ────────────────────────────────────────────────────────
        let now = frame.timestamp
        let minInterval = 1.0 / maxSendFPS
        guard now - lastSentAt >= minInterval else { return }
        lastSentAt = now

        guard let client = wsClient, let enc = encoder else { return }

        // ── Encode + send (off the delegate queue) ───────────────────────────
        let capturedFrame = frame          // ARFrame is thread-safe
        Task.detached(priority: .userInitiated) {
            guard await client.isConnected else { return }
            guard let packet = enc.encode(frame: capturedFrame) else { return }
            await client.send(packet)

            // FPS counter
            await MainActor.run {
                self.sentCount += 1
                if self.fpsWindowStart == 0 { self.fpsWindowStart = now }
                let elapsed = now - self.fpsWindowStart
                if elapsed >= 1.0 {
                    self.sentFPS = Double(self.sentCount) / elapsed
                    self.sentCount = 0
                    self.fpsWindowStart = now
                }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.statusText = "ARKit error: \(error.localizedDescription)"
        }
    }
}
