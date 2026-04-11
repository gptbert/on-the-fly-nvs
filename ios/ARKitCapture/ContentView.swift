import SwiftUI

struct ContentView: View {

    @State private var manager = ARCaptureManager()

    // Persisted settings
    @AppStorage("serverHost") private var serverHost = ""
    @AppStorage("serverPort") private var serverPort = "9000"
    @AppStorage("downsampling") private var downsampling = "1.5"
    @AppStorage("jpegQuality") private var jpegQuality  = "0.5"
    @AppStorage("maxFPS")      private var maxFPS        = "15"

    @State private var showSettings = false

    // Detect landscape vs portrait
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    private var isLandscape: Bool { verticalSizeClass == .compact }

    var body: some View {
        ZStack {
            // ── Camera preview (full screen) ──────────────────────────────────
            ARViewContainer(manager: manager)
                .ignoresSafeArea()

            if isLandscape {
                landscapeLayout
            } else {
                portraitLayout
            }
        }
        .onAppear {
            applySettings()
            manager.startARSession()
        }
        .onDisappear {
            manager.stopARSession()
            manager.disconnect()
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(
                downsampling: $downsampling,
                jpegQuality:  $jpegQuality,
                maxFPS:       $maxFPS
            )
            .onDisappear { applySettings() }
        }
    }

    // MARK: – Landscape layout
    // Controls float on the right edge so the full 16:9 frame is visible.
    private var landscapeLayout: some View {
        HStack {
            Spacer()
            VStack(spacing: 14) {
                trackingBadge

                Spacer()

                statusRow
                    .frame(maxWidth: 220)

                serverFields
                    .frame(maxWidth: 220)

                connectButton
                    .frame(maxWidth: 220)

                Button { showSettings = true } label: {
                    Label("Settings", systemImage: "gearshape")
                        .frame(maxWidth: 220)
                }
                .buttonStyle(.bordered)
            }
            .padding()
            .frame(width: 240)
            .background(.ultraThinMaterial)
        }
    }

    // MARK: – Portrait layout
    // Controls dock at the bottom.
    private var portraitLayout: some View {
        VStack {
            HStack {
                Spacer()
                trackingBadge
                    .padding()
            }
            Spacer()
            VStack(spacing: 12) {
                statusRow
                serverFields
                HStack(spacing: 10) {
                    connectButton
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding()
            .background(.ultraThinMaterial)
        }
    }

    // MARK: – Shared sub-views

    private var trackingBadge: some View {
        Text(manager.trackingState)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.ultraThinMaterial, in: Capsule())
    }

    private var statusRow: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(manager.isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(manager.statusText)
                .font(.caption).lineLimit(1)
            Spacer()
            if manager.isConnected {
                Text(String(format: "%.0f fps", manager.sentFPS))
                    .font(.caption.monospacedDigit())
            }
        }
        .foregroundStyle(.primary)
    }

    private var serverFields: some View {
        HStack(spacing: 8) {
            TextField("Server IP / hostname", text: $serverHost)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            TextField("Port", text: $serverPort)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numberPad)
                .frame(width: 68)
        }
    }

    private var connectButton: some View {
        Button { toggleConnection() } label: {
            Label(
                manager.isConnected ? "Disconnect" : "Connect",
                systemImage: manager.isConnected ? "wifi.slash" : "wifi"
            )
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .tint(manager.isConnected ? .red : .blue)
    }

    // MARK: – Helpers

    private func applySettings() {
        manager.downsampling = Float(downsampling)      ?? 1.5
        manager.jpegQuality  = CGFloat(Double(jpegQuality) ?? 0.5)
        manager.maxSendFPS   = Double(maxFPS)           ?? 15
    }

    private func toggleConnection() {
        if manager.isConnected {
            manager.disconnect()
        } else {
            guard !serverHost.isEmpty, let port = Int(serverPort) else { return }
            applySettings()
            manager.connect(host: serverHost, port: port)
        }
    }
}

// MARK: – Settings sheet
struct SettingsSheet: View {
    @Binding var downsampling: String
    @Binding var jpegQuality:  String
    @Binding var maxFPS:       String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Image") {
                    LabeledContent("Downsample factor") {
                        TextField("1.5", text: $downsampling)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                    }
                    LabeledContent("JPEG quality (0–1)") {
                        TextField("0.5", text: $jpegQuality)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                    }
                }
                Section("Streaming") {
                    LabeledContent("Max FPS") {
                        TextField("15", text: $maxFPS)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                    }
                }
                Section {
                    Text("Camera captures at 1920×1080. "
                       + "Downsample 1.5 → 1280×720; 2.0 → 960×540.\n"
                       + "Match server DOWNSAMPLING env var.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
