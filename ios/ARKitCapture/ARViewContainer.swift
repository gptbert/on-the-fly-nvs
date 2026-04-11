import SwiftUI
import ARKit

/// Bridges ARSCNView into SwiftUI, showing only the live camera feed.
/// The view is locked to portrait; ARSCNView renders the camera texture
/// and applies the correct orientation transform automatically.
struct ARViewContainer: UIViewRepresentable {
    let manager: ARCaptureManager

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = manager.arSession
        view.automaticallyUpdatesLighting = false
        view.rendersContinuously = false   // camera feed updates on demand
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
