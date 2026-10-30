import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

/// ScreenCaptureKit capture of a display rectangle, delivered to a FrameStore.
///
/// Start/update/stop run one at a time on `ops` (blocking on SCK's completion handlers), so
/// overlapping requests from the client cannot interleave. Samples arrive on `samples`, where
/// the luma plane (or BGRA) is copied out at once so SCK's surface pool never runs dry.
final class ScreenCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let frames: FrameStore
    private let ops = DispatchQueue(label: "com.osujev.hands.capture.ops")
    private let samples = DispatchQueue(label: "com.osujev.hands.capture.samples", qos: .userInteractive)
    private let lock = UnfairLock()
    private var stream: SCStream?  // lock
    private var current: CaptureConfig?  // lock
    private var displayFrame = CGRect.zero  // lock, global points
    private var configId: UInt32 = 0  // lock
    private var counts = Counts()  // lock
    private var seq: UInt64 = 0  // samples queue only

    var onStopped: ((String) -> Void)?

    struct Counts {
        var complete = 0, idle = 0, other = 0, failed = 0
        var lastPtsNanos: Int64 = 0
    }

    init(frames: FrameStore) {
        self.frames = frames
    }

    var config: CaptureConfig? { lock.with { current } }

    func start(_ cfg: CaptureConfig, reply: @escaping Reply) {
        ops.async { reply(self.startNow(cfg)) }
    }

    func update(_ cfg: CaptureConfig, reply: @escaping Reply) {
        ops.async { reply(self.updateNow(cfg)) }
    }

    func stop(reply: @escaping Reply) {
        ops.async {
            self.stopNow()
            reply(.success(self.describe()))
        }
    }

    func describe() -> [String: Any] {
        lock.with {
            var d: [String: Any] = [
                "running": stream != nil, "config_id": Int(configId),
                "frames": ["complete": counts.complete, "idle": counts.idle, "other": counts.other, "failed": counts.failed],
                "last_pts_ns": counts.lastPtsNanos,
            ]
            if let c = current {
                d["config"] = c.json
                d["display_frame"] = rectArray(displayFrame)
                d["rect"] = rectArray(c.rect ?? displayFrame)
            }
            return d
        }
    }

    // MARK: ops queue

    private func startNow(_ cfg: CaptureConfig) -> Result<[String: Any], CommandError> {
        stopNow()
        let content: SCShareableContent
        switch Self.shareableContent(onScreenOnly: true) {
        case .success(let c): content = c
        case .failure(let e): return .failure(e)
        }
        let display: SCDisplay
        if let id = cfg.display {
            guard let d = content.displays.first(where: { $0.displayID == id }) else {
                return .failure(CommandError("no display with id \(id)"))
            }
            display = d
        } else if let rect = cfg.rect {
            let center = CGPoint(x: rect.midX, y: rect.midY)
            guard let d = content.displays.first(where: { $0.frame.contains(center) }) else {
                return .failure(CommandError("rect \(rectArray(rect)) is not on any display"))
            }
            display = d
        } else {
            guard let d = content.displays.first(where: { $0.displayID == CGMainDisplayID() }) ?? content.displays.first else {
                return .failure(CommandError("no displays to capture"))
            }
            display = d
        }
        let filter: SCContentFilter
        if let id = cfg.window {
            guard let w = content.windows.first(where: { $0.windowID == id }) else {
                return .failure(CommandError("no on-screen window with id \(id)"))
            }
            filter = SCContentFilter(display: display, including: [w])
        } else {
            // OsuHands' own windows (the cursor overlay) are never captured: the model must not see them.
            let own = content.applications.filter { $0.processID == getpid() }
            filter = SCContentFilter(display: display, excludingApplications: own, exceptingWindows: [])
        }
        let configuration: SCStreamConfiguration
        switch Self.configuration(cfg, displayFrame: display.frame) {
        case .success(let c): configuration = c
        case .failure(let e): return .failure(e)
        }
        let s = SCStream(filter: filter, configuration: configuration, delegate: self)
        do {
            try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: samples)
        } catch {
            return .failure(CommandError("addStreamOutput: \(error.localizedDescription)"))
        }
        // Publish the new config id before the first sample can arrive.
        lock.with {
            stream = s
            current = cfg
            displayFrame = display.frame
            configId &+= 1
            counts = Counts()
        }
        let started = waitFor(10, { (done: @escaping (Outcome) -> Void) in s.startCapture { done(Outcome(error: $0)) } })
        if started == nil || started?.error != nil {
            lock.with {
                stream = nil
                current = nil
            }
            s.stopCapture(completionHandler: nil)
            let why = started?.error?.localizedDescription ?? "timed out"
            return .failure(CommandError("startCapture: \(why) (is Screen Recording allowed for OsuHands?)"))
        }
        Log.info("capture started: \(cfg.json) on display \(display.displayID)")
        return .success(describe())
    }

    private func updateNow(_ cfg: CaptureConfig) -> Result<[String: Any], CommandError> {
        guard let state = lock.with({ () -> (SCStream, CaptureConfig, CGRect)? in
            guard let s = stream, let c = current else { return nil }
            return (s, c, displayFrame)
        }) else { return .failure(CommandError("capture is not running")) }
        let (s, old, frame) = state
        let sameDisplay = cfg.rect.map { frame.contains(CGPoint(x: $0.midX, y: $0.midY)) } ?? true
        if cfg.display != old.display || cfg.window != old.window || cfg.format != old.format || !sameDisplay {
            return startNow(cfg)
        }
        let configuration: SCStreamConfiguration
        switch Self.configuration(cfg, displayFrame: frame) {
        case .success(let c): configuration = c
        case .failure(let e): return .failure(e)
        }
        guard let updated = waitFor(10, { (done: @escaping (Outcome) -> Void) in
            s.updateConfiguration(configuration) { done(Outcome(error: $0)) }
        }) else { return .failure(CommandError("updateConfiguration timed out")) }
        if let error = updated.error { return .failure(CommandError("updateConfiguration: \(error.localizedDescription)")) }
        // Frames already in flight may still use the old config for a frame or two.
        lock.with {
            current = cfg
            configId &+= 1
        }
        Log.info("capture updated: \(cfg.json)")
        return .success(describe())
    }

    private func stopNow() {
        guard let s = lock.with({ () -> SCStream? in
            let s = stream
            stream = nil
            current = nil
            return s
        }) else { return }
        _ = waitFor(5, { (done: @escaping (Outcome) -> Void) in s.stopCapture { done(Outcome(error: $0)) } })
        try? s.removeStreamOutput(self, type: .screen)
        Log.info("capture stopped")
    }

    static func shareableContent(onScreenOnly: Bool) -> Result<SCShareableContent, CommandError> {
        let got = waitFor(10, { (done: @escaping ((SCShareableContent?, Error?)) -> Void) in
            SCShareableContent.getExcludingDesktopWindows(true, onScreenWindowsOnly: onScreenOnly) { done(($0, $1)) }
        })
        guard let got else { return .failure(CommandError("listing shareable content timed out")) }
        guard let content = got.0 else {
            let why = got.1?.localizedDescription ?? "unknown error"
            return .failure(CommandError("cannot list screen content: \(why) (is Screen Recording allowed for OsuHands?)"))
        }
        return .success(content)
    }

    private static let black = CGColor(gray: 0, alpha: 1)  // the property is assign (unretained): keep it alive

    private static func configuration(_ cfg: CaptureConfig, displayFrame: CGRect) -> Result<SCStreamConfiguration, CommandError> {
        let rect = cfg.rect ?? displayFrame
        guard displayFrame.contains(rect.insetBy(dx: 0.5, dy: 0.5)) else {
            return .failure(CommandError("rect \(rectArray(rect)) is not inside display \(rectArray(displayFrame))"))
        }
        let c = SCStreamConfiguration()
        c.sourceRect = rect.offsetBy(dx: -displayFrame.minX, dy: -displayFrame.minY)  // display-local points
        c.width = cfg.width
        c.height = cfg.height
        c.scalesToFit = true
        c.preservesAspectRatio = false  // frame pixel = linear map of rect, never letterboxed
        c.minimumFrameInterval = cfg.fps > 0 ? CMTime(seconds: 1 / cfg.fps, preferredTimescale: 1_000_000_000) : .zero
        c.queueDepth = cfg.queueDepth
        c.showsCursor = false
        c.capturesAudio = false
        c.backgroundColor = black
        c.streamName = "OsuHands"
        switch cfg.resolution {
        case "best": c.captureResolution = .best
        case "nominal": c.captureResolution = .nominal
        default: c.captureResolution = .automatic
        }
        if cfg.colorSpace == "srgb" { c.colorSpaceName = CGColorSpace.sRGB }
        if cfg.format == Wire.bgra {
            c.pixelFormat = kCVPixelFormatType_32BGRA
        } else {
            c.pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            c.colorMatrix = cfg.matrix == "601" ? CGDisplayStream.yCbCrMatrix_ITU_R_601_4 : CGDisplayStream.yCbCrMatrix_ITU_R_709_2
        }
        return .success(c)
    }

    // MARK: SCStreamOutput, SCStreamDelegate

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }
        let recv = HostClock.nowNanos()
        guard let infos = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = infos.first,
              let rawStatus = info[.status] as? Int,
              let status = SCFrameStatus(rawValue: rawStatus) else { return }
        let (mine, cid, format) = lock.with { (self.stream === stream, configId, current?.format ?? Wire.luma) }
        guard mine else { return }
        let pts = Self.hostNanos(CMSampleBufferGetPresentationTimeStamp(sampleBuffer), clock: stream.synchronizationClock)
        let shown = (info[.displayTime] as? UInt64).map { Int64(HostClock.nanos($0)) } ?? 0
        switch status {
        case .complete, .started:
            guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let b = frames.checkout()
            guard Self.copy(pixels, into: b) else {
                frames.recycle(b)
                lock.with { counts.failed += 1 }
                return
            }
            seq &+= 1
            b.flags = 0
            b.seq = seq
            b.config = cid
            b.ptsNanos = pts
            b.displayNanos = shown
            b.recvNanos = recv
            frames.publish(b)
            lock.with {
                counts.complete += 1
                counts.lastPtsNanos = pts
            }
        case .idle:
            seq &+= 1
            frames.publishIdle(seq: seq, config: cid, format: format, pts: pts, shown: shown, recv: recv)
            lock.with { counts.idle += 1 }
        default:
            lock.with { counts.other += 1 }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        let mine = lock.with { () -> Bool in
            guard self.stream === stream else { return false }
            self.stream = nil
            current = nil
            return true
        }
        Log.info("capture stopped by the system: \(error.localizedDescription)")
        if mine { onStopped?(error.localizedDescription) }
    }

    /// Copies the luma plane (420f) or the BGRA plane into `b`, dropping any row padding.
    private static func copy(_ pb: CVPixelBuffer, into b: FrameBuffer) -> Bool {
        guard CVPixelBufferLockBaseAddress(pb, .readOnly) == kCVReturnSuccess else { return false }
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let planar = CVPixelBufferIsPlanar(pb)
        guard let base = planar ? CVPixelBufferGetBaseAddressOfPlane(pb, 0) : CVPixelBufferGetBaseAddress(pb) else { return false }
        let w = planar ? CVPixelBufferGetWidthOfPlane(pb, 0) : CVPixelBufferGetWidth(pb)
        let h = planar ? CVPixelBufferGetHeightOfPlane(pb, 0) : CVPixelBufferGetHeight(pb)
        let stride = planar ? CVPixelBufferGetBytesPerRowOfPlane(pb, 0) : CVPixelBufferGetBytesPerRow(pb)
        let bpp = planar ? 1 : 4
        let row = w * bpp
        guard row <= stride else { return false }
        b.reserve(row * h)
        if stride == row {
            memcpy(b.bytes, base, row * h)
        } else {
            for y in 0..<h { memcpy(b.bytes + y * row, base + y * stride, row) }
        }
        b.length = row * h
        b.width = w
        b.height = h
        b.format = planar ? Wire.luma : Wire.bgra
        return true
    }

    /// A sample time as ns on mach_absolute_time. SCK stamps frames on its synchronization clock,
    /// which is the host time clock; convert through it anyway in case that ever changes.
    private static func hostNanos(_ t: CMTime, clock: CMClock?) -> Int64 {
        guard t.isValid, t.isNumeric else { return 0 }
        let host = CMClockGetHostTimeClock()
        var hostTime = t
        if let clock, !CFEqual(clock, host) { hostTime = CMSyncConvertTime(t, from: clock, to: host) }
        return Int64(HostClock.nanos(CMClockConvertHostTimeToSystemUnits(hostTime)))
    }
}
