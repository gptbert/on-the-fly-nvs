import SwiftUI
import ARKit
import SceneKit

/// Bridges ARSCNView (UIKit) into SwiftUI.
/// Shows the live camera feed; no 3-D content is added.
struct ARViewContainer: UIViewRepresentable {
    let manager: ARCaptureManager

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = manager.arSession
        view.automaticallyUpdatesLighting = false
        view.scene = SCNScene()
        // Prevent the scene-kit render loop from fighting with our delegate
        view.rendersContinuously = false
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
