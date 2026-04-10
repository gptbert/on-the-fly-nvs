import Foundation

/// Thin wrapper around URLSessionWebSocketTask.
/// All public methods are safe to call from any thread.
actor WebSocketClient {
    private var task: URLSessionWebSocketTask?
    private let url: URL
    private let session: URLSession

    var isConnected: Bool { task?.state == .running }

    init(url: URL) {
        self.url = url
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        self.session = URLSession(configuration: config)
    }

    // MARK: – Connection lifecycle

    func connect() async {
        disconnect()
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        // Start a receive loop so the system keeps the connection alive
        // and reports close/error events.
        receiveLoop(task: t)
    }

    func disconnect() {
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    // MARK: – Sending

    func send(_ data: Data) async {
        guard let t = task, t.state == .running else { return }
        do {
            try await t.send(.data(data))
        } catch {
            print("[WS] send error: \(error.localizedDescription)")
        }
    }

    // MARK: – Private

    private func receiveLoop(task: URLSessionWebSocketTask) {
        Task {
            do {
                _ = try await task.receive()   // we don't expect server messages
                receiveLoop(task: task)
            } catch {
                // Connection closed or errored; nothing to do – isConnected will
                // reflect the task state naturally.
            }
        }
    }
}
