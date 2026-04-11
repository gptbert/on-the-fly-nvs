import SwiftUI

struct ContentView: View {

    @State private var manager = ARCaptureManager()

    @AppStorage("serverHost") private var serverHost = ""
    @AppStorage("serverPort") private var serverPort = "9000"
    @AppStorage("downsampling") private var downsampling = "1.5"
    @AppStorage("jpegQuality") private var jpegQuality  = "0.5"
    @AppStorage("maxFPS")      private var maxFPS        = "15"

    @State private var showSettings = false

    var body: some View {
        ZStack(alignment: .bottom) {

            // ── Camera preview (full screen) ──────────────────────────────────
            ARViewContainer(manager: manager)
                .ignoresSafeArea()

            // ── Tracking badge (top-right) ────────────────────────────────────
            VStack {
                HStack {
                    Spacer()
                    Text(manager.trackingState)
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.trailing)
                }
                Spacer()
            }
            .padding(.top)

            // ── Bottom control dock ───────────────────────────────────────────
            VStack(spacing: 12) {

                // Status row
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

                // Server address
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

                // Connect + Settings
                HStack(spacing: 10) {
                    Button { toggleConnection() } label: {
                        Label(
                            manager.isConnected ? "Disconnect" : "Connect",
                            systemImage: manager.isConnected ? "wifi.slash" : "wifi"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(manager.isConnected ? .red : .blue)

                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding()
            .background(.ultraThinMaterial)
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

    private func applySettings() {
        manager.downsampling = Float(downsampling)         ?? 1.5
        manager.jpegQuality  = CGFloat(Double(jpegQuality) ?? 0.5)
        manager.maxSendFPS   = Double(maxFPS)              ?? 15
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
                    Text("Sensor captures 1920×1080 (16:9 landscape) and rotates "
                       + "to portrait (1080×1920).\n"
                       + "Downsample 1.5 → 720×1280 · 2.0 → 540×960.\n"
                       + "Set server DOWNSAMPLING to match.")
                        .font(.caption).foregroundStyle(.secondary)
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
