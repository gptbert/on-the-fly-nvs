import ARKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers

/// Encodes an ARFrame (image + pose) into the server wire protocol:
///   [4-byte big-endian UInt32: JSON length][UTF-8 JSON][JPEG bytes]
///
/// JSON fields:
///   transform  [Float × 16]  row-major 4×4 camera-to-world (ARKit convention)
///   fx, fy     Float         focal lengths at the *sent* resolution
///   cx, cy     Float         principal point at the *sent* resolution
struct FrameEncoder {
    /// Downsample factor applied to both image and intrinsics.
    let downsampling: Float
    /// JPEG quality [0, 1].
    let jpegQuality: CGFloat

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    init(downsampling: Float = 1.5, jpegQuality: CGFloat = 0.5) {
        self.downsampling = downsampling
        self.jpegQuality = jpegQuality
    }

    func encode(frame: ARFrame) -> Data? {
        let pixelBuffer = frame.capturedImage
        let transform   = frame.camera.transform      // simd_float4x4, C2W
        let intr        = frame.camera.intrinsics     // simd_float3x3, column-major

        // ── 1. Image ──────────────────────────────────────────────────────────
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        if downsampling != 1.0 {
            let s = CGFloat(1.0 / downsampling)
            ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: s, y: s))
        }
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        let imageData = NSMutableData()
        guard
            let dest = CGImageDestinationCreateWithData(
                imageData, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(
            dest, cgImage,
            [kCGImageDestinationLossyCompressionQuality: jpegQuality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }

        // ── 2. Pose (row-major C2W) ──────────────────────────────────────────
        // simd_float4x4 stores columns; we need rows.
        let c = transform.columns
        let transformArray: [Float] = [
            c.0.x, c.1.x, c.2.x, c.3.x,
            c.0.y, c.1.y, c.2.y, c.3.y,
            c.0.z, c.1.z, c.2.z, c.3.z,
            c.0.w, c.1.w, c.2.w, c.3.w,
        ]

        // ── 3. Intrinsics (scaled for downsampling) ──────────────────────────
        // ARKit simd_float3x3 column-major: [0][0]=fx, [1][1]=fy, [2][0]=cx, [2][1]=cy
        let ds = downsampling
        let fx = intr[0][0] / ds
        let fy = intr[1][1] / ds
        let cx = intr[2][0] / ds
        let cy = intr[2][1] / ds

        // ── 4. JSON header ────────────────────────────────────────────────────
        let jsonObject: [String: Any] = [
            "transform": transformArray,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        ]
        guard let jsonBytes = try? JSONSerialization.data(withJSONObject: jsonObject) else {
            return nil
        }

        // ── 5. Wire message: [4-byte length][JSON][JPEG] ─────────────────────
        var lengthBE = UInt32(jsonBytes.count).bigEndian
        var packet = Data(capacity: 4 + jsonBytes.count + imageData.length)
        packet.append(Data(bytes: &lengthBE, count: 4))
        packet.append(jsonBytes)
        packet.append(imageData as Data)
        return packet
    }
}
