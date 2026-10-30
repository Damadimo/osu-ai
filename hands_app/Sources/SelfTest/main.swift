// hands_selftest: the OsuHands socket server and real-time input scheduler with a fake capture
// source and a dry-run input poster. It never reads the screen or posts events, so it needs no
// permissions; tests/test_hands_client.py drives it with the Python client.
//
//   hands_selftest --socket PATH
//
// Extra commands: _fake_front {bundle} (pretend another app is frontmost),
// _fake_log {} (the input actions "posted" so far, with execution times; clears the log),
// _fake_idle {every} (make every Nth frame an idle frame; 0: none).
import Foundation

final class DryRunPoster: InputPoster {
    private let lock = UnfairLock()
    private var front = "sh.ppy.osu.lazer"
    private var target = "sh.ppy.osu.lazer"
    private var log: [[Any]] = []

    func inputAllowed() -> Bool { lock.with { front == target } }

    func postMove(x: Double, y: Double, dx: Double, dy: Double) { record(["move", 0, x, y]) }

    func postKey(_ key: UInt16, down: Bool) { record([down ? "down" : "up", Int(key), 0.0, 0.0]) }
    func postMouse(_ button: UInt16, down: Bool, x: Double, y: Double) { record([down ? "mdown" : "mup", Int(button), x, y]) }

    private func record(_ entry: [Any]) {
        let t = HostClock.nowNanos()
        lock.with { log.append(entry + [t]) }
    }

    var targetBundle: String {
        get { lock.with { target } }
        set { lock.with { target = newValue } }
    }

    func setFront(_ bundle: String) { lock.with { front = bundle } }

    func info() -> [String: Any] {
        lock.with { ["bundle_id": front, "name": front, "pid": 0, "target_bundle": target, "allowed": front == target] }
    }

    func drain() -> [[Any]] {
        lock.with {
            let out = log
            log.removeAll()
            return out
        }
    }
}

/// Frames with a known pattern: byte (row y, byte x) = (x + 2y + seq) & 255, then the first
/// 8 bytes overwritten with seq (u64 LE). pts is 4 ms before delivery.
final class FakeBackend: Backend {
    let name = "selftest"
    let poster = DryRunPoster()
    var onEvent: (([String: Any]) -> Void)?
    private let frames: FrameStore
    private let queue = DispatchQueue(label: "selftest.frames", qos: .userInteractive)
    private let lock = UnfairLock()
    private var current: CaptureConfig?  // lock
    private var configId: UInt32 = 0  // lock
    private var timer: DispatchSourceTimer?  // queue
    private var seq: UInt64 = 0  // queue
    private var idleEvery: UInt64 = 0  // queue

    init(frames: FrameStore) { self.frames = frames }

    var targetBundle: String {
        get { poster.targetBundle }
        set {
            poster.targetBundle = newValue
            emitFrontmost()
        }
    }

    var captureConfig: CaptureConfig? { lock.with { current } }

    func permissions() -> [String: Any] { ["screen_capture": false, "accessibility": false, "post_event": false] }

    func frontmost() -> [String: Any] { poster.info() }

    func captureStatus() -> [String: Any] {
        lock.with {
            var d: [String: Any] = ["running": current != nil, "config_id": Int(configId)]
            if let c = current { d["config"] = c.json }
            return d
        }
    }

    func requestPermissions(_ args: [String: Any], reply: @escaping Reply) { reply(.success(permissions())) }

    func displays(reply: @escaping Reply) {
        reply(.success(["displays": [[
            "id": 1, "main": true, "frame": [0.0, 0.0, 1512.0, 982.0], "pixels": [3024, 1964], "scale": 2.0,
            "refresh_hz": 120.0, "name": "Fake Display", "backing_scale": 2.0, "max_fps": 120,
            "safe_area_insets": ["top": 32.0, "left": 0.0, "bottom": 0.0, "right": 0.0],
        ] as [String: Any]]]))
    }

    func windows(_ args: [String: Any], reply: @escaping Reply) {
        reply(.success(["source": args["source"] as? String ?? "sck", "windows": [
            ["id": 7, "title": "Finder", "app": "Finder", "bundle_id": "com.apple.finder", "pid": 400,
             "frame": [0.0, 0.0, 1512.0, 982.0], "layer": 0, "on_screen": true, "active": false],
            ["id": 42, "title": "osu!", "app": "osu!", "bundle_id": "sh.ppy.osu.lazer", "pid": 500,
             "frame": [0.0, 32.0, 1512.0, 950.0], "layer": 0, "on_screen": true, "active": true],
        ] as [[String: Any]]]))
    }

    func startCapture(_ config: CaptureConfig, reply: @escaping Reply) {
        queue.async {
            self.timer?.cancel()
            let id = self.lock.with { () -> UInt32 in
                self.current = config
                self.configId &+= 1
                return self.configId
            }
            let timer = DispatchSource.makeTimerSource(flags: .strict, queue: self.queue)
            let interval = config.fps > 0 ? 1 / config.fps : 1.0 / 120
            timer.schedule(deadline: .now(), repeating: interval, leeway: .microseconds(200))
            timer.setEventHandler { self.produce(config, id) }
            timer.resume()
            self.timer = timer
            reply(.success(self.captureStatus().merging([
                "display_frame": [0.0, 0.0, 1512.0, 982.0], "rect": rectArray(config.rect ?? CGRect(x: 0, y: 0, width: 1512, height: 982)),
            ]) { $1 }))
        }
    }

    func updateCapture(_ config: CaptureConfig, reply: @escaping Reply) { startCapture(config, reply: reply) }

    func stopCapture(reply: @escaping Reply) {
        queue.async {
            self.timer?.cancel()
            self.timer = nil
            self.lock.with { self.current = nil }
            reply(.success(self.captureStatus()))
        }
    }

    private func produce(_ c: CaptureConfig, _ id: UInt32) {
        seq &+= 1
        let now = HostClock.nowNanos()
        if idleEvery > 0 && seq % idleEvery == 0 {
            frames.publishIdle(seq: seq, config: id, format: c.format, pts: now - 4_000_000, shown: now - 4_000_000, recv: now)
            return
        }
        let bpp = c.format == Wire.bgra ? 4 : 1
        let row = c.width * bpp
        let b = frames.checkout()
        b.reserve(row * c.height)
        let p = b.bytes.assumingMemoryBound(to: UInt8.self)
        for y in 0..<c.height {
            for x in 0..<row { p[y * row + x] = UInt8(truncatingIfNeeded: x + 2 * y + Int(seq)) }
        }
        withUnsafeBytes(of: seq.littleEndian) { b.bytes.copyMemory(from: $0.baseAddress!, byteCount: 8) }
        b.length = row * c.height
        b.width = c.width
        b.height = c.height
        b.format = c.format
        b.flags = 0
        b.seq = seq
        b.config = id
        b.ptsNanos = now - 4_000_000
        b.displayNanos = now - 4_000_000
        b.recvNanos = now
        frames.publish(b)
    }

    func extra(_ cmd: String, _ args: [String: Any], reply: @escaping Reply) -> Bool {
        switch cmd {
        case "_fake_front":
            poster.setFront(args["bundle"] as? String ?? "")
            emitFrontmost()
            reply(.success(poster.info()))
        case "open_settings":
            reply(.success(["opened": "nothing (selftest)"]))
        case "_fake_log":
            reply(.success(["log": poster.drain()]))
        case "_fake_idle":
            let every = (args["every"] as? NSNumber)?.uint64Value ?? 0
            queue.async {
                self.idleEvery = every
                reply(.success(["every": Int(every)]))
            }
        default:
            return false
        }
        return true
    }

    private func emitFrontmost() {
        var e = poster.info()
        e["event"] = "frontmost"
        onEvent?(e)
    }

    func quit() { exit(0) }
}

guard let i = CommandLine.arguments.firstIndex(of: "--socket"), i + 1 < CommandLine.arguments.count else {
    FileHandle.standardError.write(Data("usage: hands_selftest --socket PATH\n".utf8))
    exit(2)
}
signal(SIGPIPE, SIG_IGN)
let frames = FrameStore()
let backend = FakeBackend(frames: frames)
let server = Server(path: CommandLine.arguments[i + 1], backend: backend, frames: frames, poster: backend.poster)
do {
    try server.start()
} catch {
    FileHandle.standardError.write(Data("hands_selftest: \(error)\n".utf8))
    exit(1)
}
var stops: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler {
        server.shutdown()
        exit(0)
    }
    source.resume()
    stops.append(source)
}
dispatchMain()
