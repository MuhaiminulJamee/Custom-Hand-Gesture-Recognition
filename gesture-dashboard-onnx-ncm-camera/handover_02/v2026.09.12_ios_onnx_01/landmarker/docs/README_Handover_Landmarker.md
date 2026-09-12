# Stage 1: Hand Landmarker Package (iOS)

This package contains the hand detector model and the corrected Swift feature engineering pipeline.

## Contents
- `models/hand_landmarker.task`: Google MediaPipe Hand Landmarker model (21 hand landmarks).
- `swift/GestureFeatureExtractor.swift`: Mathematical feature extractor producing the 76-D input vector for the ONNX model.
- `docs/geometry.py`: Original Python reference geometry implementation.
- `contracts/LANDMARKER_RELEASE_MANIFEST.json`: Release metadata and checksums.

## iOS Setup Guide

### 1. MediaPipe Tasks Vision iOS SDK
Install the official MediaPipe Tasks Vision SDK via CocoaPods or Swift Package Manager:
```ruby
# Podfile
pod 'MediaPipeTasksVision'
```
Or SPM URL: `https://github.com/google-ai-edge/mediapipe`

### 2. Camera Configuration & Mirroring (CRITICAL)
- **Disable Mirroring:** On iOS front cameras, `isVideoMirrored` defaults to `true`. You **MUST set `isVideoMirrored = false`**. If left mirrored, `Move Left` and `Move Right` will be inverted!
```swift
if let connection = videoOutput.connection(with: .video) {
    if connection.isVideoMirroringSupported {
        connection.isVideoMirrored = false // REQUIRED: Must be unmirrored
    }
    connection.videoOrientation = .portrait
}
```
- **Orientation:** Ensure video frames passed to MediaPipe use `.up` orientation matching physical device orientation.
- **Coordinates:** MediaPipe provides normalized `(x, y)` in `[0.0, 1.0]` where `(0.0, 0.0)` is **top-left**.

### 3. Running Feature Extraction
```swift
import simd

// 1. Get 21 landmarks from MediaPipe HandLandmarkerResult
let landmarks: [SIMD2<Float>] = result.landmarks[0].map { landmark in
    SIMD2<Float>(landmark.x, landmark.y)
}

// 2. Extract the exact 76 features
do {
    let features76 = try GestureFeatureExtractor.landmarksToFeature(landmarks)
    // Pass features76 ([Float], count = 76) to ONNX Runtime
} catch {
    print("Geometry error: \(error)")
}
```

### 4. Parity Self-Test
You can verify mathematical parity with the Python training backend directly in Xcode by running:
```swift
let test = GestureFeatureExtractor.runParitySelfTest()
print("Parity test passed: \(test.passed), max diff: \(test.maxFeatureDiff)")
```
If `test.passed == true`, your Swift feature extraction matches Python to within $10^{-5}$.
