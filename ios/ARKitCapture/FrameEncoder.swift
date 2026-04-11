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
///
/// Portrait rotation:
///   ARKit always provides the sensor image in landscape orientation
///   (phone's "up" direction = left edge of the image).  When rotateToPortrait
///   is true the encoder rotates 90° CW so the image appears upright, and
///   remaps the intrinsics accordingly:
///     fx_new = fy_old,  cx_new = H_old – cy_old
///     fy_new = fx_old,  cy_new = cx_old
struct FrameEncoder {
    let downsampling: Float
    let jpegQuality: CGFloat
    /// Rotate sensor (landscape) image 90° CW to produce a portrait image.
    let rotateToPortrait: Bool

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    init(downsampling: Float = 1.5, jpegQuality: CGFloat = 0.5, rotateToPortrait: Bool = true) {
        self.downsampling     = downsampling
        self.jpegQuality      = jpegQuality
        self.rotateToPortrait = rotateToPortrait
    }

    func encode(frame: ARFrame) -> Data? {
        let pixelBuffer = frame.capturedImage        // always landscape from sensor
        let transform   = frame.camera.transform     // simd_float4x4, C2W
        let intr        = frame.camera.intrinsics    // simd_float3x3, column-major
        // intr[col][row]: [0][0]=fx, [1][1]=fy, [2][0]=cx, [2][1]=cy

        // ── 1. Image ──────────────────────────────────────────────────────────
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Sensor native height, needed for cx_new when rotating
        let sensorH = Float(ciImage.extent.height)  // 1080 for a 1920×1080 format

        if rotateToPortrait {
            // 90° CW: phone's "up" (left edge in landscape) moves to the top.
            // .oriented(.right) applies a 90° CW correction (EXIF tag 6).
            ciImage = ciImage.oriented(.right)
        }

        if downsampling != 1.0 {
            let s = CGFloat(1.0 / downsampling)
            ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: s, y: s))
        }

        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        let imageData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            imageData, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(
            dest, cgImage,
            [kCGImageDestinationLossyCompressionQuality: jpegQuality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }

        // ── 2. Pose (row-major C2W) ──────────────────────────────────────────
        let c = transform.columns
        let transformArray: [Float] = [
            c.0.x, c.1.x, c.2.x, c.3.x,
            c.0.y, c.1.y, c.2.y, c.3.y,
            c.0.z, c.1.z, c.2.z, c.3.z,
            c.0.w, c.1.w, c.2.w, c.3.w,
        ]

        // ── 3. Intrinsics ────────────────────────────────────────────────────
        let ds = downsampling
        let fx, fy, cx, cy: Float

        if rotateToPortrait {
            // After 90° CW rotation of a W×H image:
            //   new pixel (x', y') ← old pixel (H−1−y', x')
            //   fx' = fy,  cx' = H − cy   (≈ H − cy, ignoring −1 at this scale)
            //   fy' = fx,  cy' = cx
            fx = intr[1][1] / ds          // fy_old
            fy = intr[0][0] / ds          // fx_old
            cx = (sensorH - intr[2][1]) / ds  // H_old − cy_old
            cy = intr[2][0] / ds          // cx_old
        } else {
            fx = intr[0][0] / ds
            fy = intr[1][1] / ds
            cx = intr[2][0] / ds
            cy = intr[2][1] / ds
        }

        // ── 4. JSON header ────────────────────────────────────────────────────
        let jsonObject: [String: Any] = [
            "transform": transformArray,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
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
