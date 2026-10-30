import CoreGraphics
import Darwin
import Foundation

/// A capture request. `rect` is in global display points; width x height is the output in pixels.
struct CaptureConfig {
    var rect: CGRect?  // nil: the whole display
    var display: UInt32?  // nil: the display under rect's center (the main display without a rect)
    var window: UInt32?  // capture only this CGWindowID (still cropped to rect)
    var width = 640
    var height = 480
    var fps = 120.0  // 0: every display refresh
    var queueDepth = 3
    var format = Wire.luma
    var resolution = "auto"  // auto | best | nominal
    var colorSpace = "srgb"  // srgb | display
    var matrix = "709"  // 709 | 601, luma only

    static func parse(_ a: [String: Any], base: CaptureConfig?) throws -> CaptureConfig {
        var c = base ?? CaptureConfig()
        if let v = a["rect"] {
            if v is NSNull {
                c.rect = nil
            } else if let r = numbers(v), r.count == 4, r[2] > 0, r[3] > 0 {
                c.rect = CGRect(x: r[0], y: r[1], width: r[2], height: r[3])
            } else {
                throw CommandError("rect must be [x, y, w, h] in global points, or null")
            }
        }
        if let v = a["display"] { c.display = (v as? NSNumber)?.uint32Value }
        if let v = a["window"] { c.window = (v as? NSNumber)?.uint32Value }
        if let v = a["size"] {
            guard let s = numbers(v), s.count == 2 else { throw CommandError("size must be [width, height]") }
            c.width = Int(s[0])
            c.height = Int(s[1])
        }
        if let v = a["fps"] as? NSNumber { c.fps = v.doubleValue }
        if let v = a["queue_depth"] as? NSNumber { c.queueDepth = v.intValue }
        if let v = a["format"] as? String {
            switch v {
            case "luma": c.format = Wire.luma
            case "bgra": c.format = Wire.bgra
            default: throw CommandError("format must be luma or bgra")
            }
        }
        if let v = a["resolution"] as? String {
            guard ["auto", "best", "nominal"].contains(v) else { throw CommandError("resolution must be auto, best or nominal") }
            c.resolution = v
        }
        if let v = a["color_space"] as? String {
            guard ["srgb", "display"].contains(v) else { throw CommandError("color_space must be srgb or display") }
            c.colorSpace = v
        }
        if let v = a["matrix"] as? String {
            guard ["709", "601"].contains(v) else { throw CommandError("matrix must be 709 or 601") }
            c.matrix = v
        }
        guard (16...4096).contains(c.width), (16...4096).contains(c.height), c.width % 2 == 0, c.height % 2 == 0 else {
            throw CommandError("size must be even numbers in 16...4096")
        }
        guard c.fps >= 0, c.fps <= 240 else { throw CommandError("fps must be in 0...240 (0: every refresh)") }
        guard (3...8).contains(c.queueDepth) else { throw CommandError("queue_depth must be in 3...8") }
        return c
    }

    var json: [String: Any] {
        [
            "rect": rect.map(rectArray) ?? NSNull(), "display": display.map { Int($0) } ?? NSNull(),
            "window": window.map { Int($0) } ?? NSNull(), "size": [width, height], "fps": fps,
            "queue_depth": queueDepth, "format": format == Wire.bgra ? "bgra" : "luma",
            "resolution": resolution, "color_space": colorSpace, "matrix": matrix,
        ]
    }
}

/// What differs between the app and the self-test: capture, windows, permissions, focus.
protocol Backend: AnyObject {
    var name: String { get }
    var targetBundle: String { get set }
    /// Unsolicited events for the client ("frontmost", "capture_stopped").
    var onEvent: (([String: Any]) -> Void)? { get set }
    var captureConfig: CaptureConfig? { get }
    func permissions() -> [String: Any]
    func frontmost() -> [String: Any]
    func captureStatus() -> [String: Any]
    func requestPermissions(_ args: [String: Any], reply: @escaping Reply)
    func displays(reply: @escaping Reply)
    func windows(_ args: [String: Any], reply: @escaping Reply)
    func startCapture(_ config: CaptureConfig, reply: @escaping Reply)
    func updateCapture(_ config: CaptureConfig, reply: @escaping Reply)
    func stopCapture(reply: @escaping Reply)
    /// Backend-only commands; returns false if `cmd` is unknown.
    func extra(_ cmd: String, _ args: [String: Any], reply: @escaping Reply) -> Bool
    func quit()
}

/// The socket server: one client session at a time, control commands, input, frames.
final class Server {
    let path: String
    let backend: Backend
    let frames: FrameStore
    let scheduler: InputScheduler
    private let control = DispatchQueue(label: "com.osujev.hands.control", qos: .userInitiated)
    private let lock = UnfairLock()
    private var session: Session?  // lock
    private var sessionCount = 0  // lock
    private var listenFD: Int32 = -1
    private let startedNanos = HostClock.nowNanos()

    init(path: String, backend: Backend, frames: FrameStore, poster: InputPoster) {
        self.path = path
        self.backend = backend
        self.frames = frames
        scheduler = InputScheduler(poster: poster)
    }

    var connected: Bool { lock.with { session != nil } }

    func start() throws {
        let dir = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if UnixSocket.canConnect(path) { throw CommandError("another OsuHands is already serving \(path)") }
        unlink(path)
        listenFD = try UnixSocket.serve(path)
        backend.onEvent = { [weak self] in self?.event($0) }
        let thread = Thread { self.acceptLoop() }
        thread.name = "com.osujev.hands.accept"
        thread.start()
        Log.info("listening on \(path)")
    }

    /// Releases held keys synchronously, drops the client and removes the socket.
    func shutdown() {
        scheduler.stopAndRelease()
        lock.with { session }?.kick()
        unlink(path)
    }

    private func acceptLoop() {
        while true {
            let fd = accept(listenFD, nil, nil)
            if fd < 0 {
                if errno == EINTR || errno == ECONNABORTED { continue }
                Log.info("accept failed: \(String(cString: strerror(errno)))")
                return
            }
            UnixSocket.configure(fd)
            attach(fd)
        }
    }

    private func attach(_ fd: Int32) {
        let (old, s) = lock.with { () -> (Session?, Session) in
            sessionCount += 1
            let s = Session(fd: fd, id: sessionCount, server: self)
            let old = session
            session = s
            return (old, s)
        }
        if let old {
            Log.info("session \(old.id) replaced by session \(s.id)")
            old.kick()
        }
        // Every session starts clean: nothing pending, no keys held, no capture running.
        scheduler.reset()
        frames.reset()
        if backend.captureConfig != nil { backend.stopCapture { _ in } }
        frames.setWaker { [weak s] in s?.wake() }
        s.send(hello())  // queued first, so it is the first message the client sees
        s.start()
        Log.info("session \(s.id) connected")
    }

    fileprivate func ended(_ s: Session) {
        let current = lock.with { () -> Bool in
            guard session === s else { return false }
            session = nil
            return true
        }
        Log.info("session \(s.id) ended")
        guard current else { return }
        frames.setWaker(nil)
        scheduler.reset()
        frames.reset()
        backend.stopCapture { _ in }
    }

    private func event(_ e: [String: Any]) {
        // Losing focus drops everything pending and releases held keys: stale plans must not fire on return.
        if e["event"] as? String == "frontmost", e["allowed"] as? Bool == false { scheduler.reset() }
        if e["event"] as? String == "capture_stopped" { frames.reset() }
        lock.with { session }?.send(e)
    }

    private func hello() -> [String: Any] {
        [
            "event": "hello", "version": Wire.version, "backend": backend.name, "pid": Int(getpid()),
            "now_ns": HostClock.nowNanos(), "clock": "mach_absolute_time",
            "timebase": [Int(HostClock.timebase.numer), Int(HostClock.timebase.denom)],
            "target_bundle": backend.targetBundle, "frontmost": backend.frontmost(), "permissions": backend.permissions(),
        ]
    }

    // MARK: commands

    fileprivate func command(_ data: Data, from s: Session) {
        control.async {
            guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                s.send(["ok": false, "error": "request is not a JSON object"])
                return
            }
            let id = obj["id"]
            let reply: Reply = { result in
                var out: [String: Any]
                switch result {
                case .success(let fields):
                    out = fields
                    out["ok"] = true
                case .failure(let error):
                    out = ["ok": false, "error": error.description]
                }
                if let id { out["id"] = id }
                s.send(out)
            }
            self.run(obj["cmd"] as? String ?? "", obj, reply)
        }
    }

    private func run(_ cmd: String, _ a: [String: Any], _ reply: @escaping Reply) {
        switch cmd {
        case "ping":
            reply(.success(["now_ns": HostClock.nowNanos(), "version": Wire.version]))
        case "status":
            reply(.success(status()))
        case "stats":
            reply(.success(["input": scheduler.statsJSON(reset: a["reset"] as? Bool ?? false),
                            "frames": frames.statsJSON(), "capture": backend.captureStatus()]))
        case "config":
            reply(configure(a))
        case "release":
            scheduler.reset()
            reply(.success([:]))
        case "request_permissions":
            backend.requestPermissions(a, reply: reply)
        case "displays":
            backend.displays(reply: reply)
        case "windows":
            backend.windows(a, reply: reply)
        case "capture_start":
            do { backend.startCapture(try CaptureConfig.parse(a, base: nil), reply: reply) } catch { reply(.failure(Self.wrap(error))) }
        case "capture_update":
            guard let base = backend.captureConfig else { return reply(.failure(CommandError("capture is not running"))) }
            do { backend.updateCapture(try CaptureConfig.parse(a, base: base), reply: reply) } catch { reply(.failure(Self.wrap(error))) }
        case "capture_stop":
            backend.stopCapture(reply: reply)
        case "quit":
            reply(.success([:]))
            control.asyncAfter(deadline: .now() + 0.05) { self.backend.quit() }
        default:
            if !backend.extra(cmd, a, reply: reply) { reply(.failure(CommandError("unknown command '\(cmd)'"))) }
        }
    }

    private static func wrap(_ error: Error) -> CommandError { error as? CommandError ?? CommandError("\(error)") }

    private func status() -> [String: Any] {
        [
            "version": Wire.version, "backend": backend.name, "pid": Int(getpid()), "socket": path,
            "bundle_path": Bundle.main.bundlePath, "now_ns": HostClock.nowNanos(),
            "uptime_s": Double(HostClock.nowNanos() - startedNanos) / 1e9,
            "clock": ["source": "mach_absolute_time", "numer": Int(HostClock.timebase.numer), "denom": Int(HostClock.timebase.denom)],
            "target_bundle": backend.targetBundle, "permissions": backend.permissions(), "frontmost": backend.frontmost(),
            "clamp": scheduler.clampRect.map(rectArray) ?? NSNull(), "realtime_input": scheduler.realtime,
            "capture": backend.captureStatus(), "frames": frames.statsJSON(),
        ]
    }

    private func configure(_ a: [String: Any]) -> Result<[String: Any], CommandError> {
        if let t = a["target_bundle"] as? String { backend.targetBundle = t }
        if let c = a["clamp"] {
            if c is NSNull {
                scheduler.clampRect = nil
            } else if let r = numbers(c), r.count == 4, r[2] >= 1, r[3] >= 1 {
                scheduler.clampRect = CGRect(x: r[0], y: r[1], width: r[2], height: r[3])
            } else {
                return .failure(CommandError("clamp must be [x, y, w, h] in global points, or null"))
            }
        }
        return .success([
            "target_bundle": backend.targetBundle, "clamp": scheduler.clampRect.map(rectArray) ?? NSNull(),
            "frontmost": backend.frontmost(),
        ])
    }

    // MARK: input

    fileprivate func input(_ raw: UnsafeRawBufferPointer) {
        guard raw.count % Wire.record == 0 else {
            Log.info("input message of \(raw.count) bytes is not a multiple of \(Wire.record)")
            return
        }
        let limit = HostClock.now() &+ HostClock.ticks(InputScheduler.maxAheadNanos)
        var batch: [InputAction] = []
        var rejected = 0
        for r in InputRecord.parse(raw) {
            switch r.kind {
            case Wire.cancel:
                scheduler.submit(batch)
                batch.removeAll()
                scheduler.cancel(tag: r.tag)
            case Wire.move, Wire.keyDown, Wire.keyUp, Wire.releaseAll, Wire.mouseDown, Wire.mouseUp:
                let deadline = r.atNanos > 0 ? HostClock.ticks(UInt64(r.atNanos)) : 0
                if deadline > limit || ([Wire.move, Wire.mouseDown, Wire.mouseUp].contains(r.kind) && !(r.x.isFinite && r.y.isFinite)) {
                    rejected += 1
                    continue
                }
                batch.append(InputAction(deadline: deadline, kind: r.kind, key: r.key, tag: r.tag, x: r.x, y: r.y))
            default:
                rejected += 1
            }
        }
        scheduler.submit(batch)
        if rejected > 0 { scheduler.noteRejected(rejected) }
    }
}

/// One connected client: a reader thread (commands, input) and a writer thread (replies, frames).
final class Session {
    let fd: Int32
    let id: Int
    private let server: Server
    private let cond = NSCondition()
    private var outbox: [Data] = []  // cond
    private var closing = false  // cond
    private var fdClosed = false  // cond
    private let writerDone = DispatchSemaphore(value: 0)

    init(fd: Int32, id: Int, server: Server) {
        self.fd = fd
        self.id = id
        self.server = server
    }

    func start() {
        for (name, body) in [("writer", writeLoop), ("reader", readLoop)] {
            let thread = Thread(block: body)
            thread.name = "com.osujev.hands.\(name).\(id)"
            thread.qualityOfService = .userInteractive
            thread.start()
        }
    }

    func send(_ obj: [String: Any]) {
        guard let msg = Wire.encodeJSON(obj) else {
            Log.info("session \(id): reply is not valid JSON: \(obj)")
            return
        }
        cond.lock()
        if !closing {
            outbox.append(msg)
            cond.signal()
        }
        cond.unlock()
    }

    func wake() {
        cond.lock()
        cond.signal()
        cond.unlock()
    }

    /// Ends the session: the reader sees EOF and cleans up.
    func kick() {
        cond.lock()
        if !fdClosed { Darwin.shutdown(fd, SHUT_RDWR) }
        cond.unlock()
    }

    private func readLoop() {
        var head = [UInt8](repeating: 0, count: 4)
        var body = [UInt8](repeating: 0, count: 4096)
        while head.withUnsafeMutableBytes({ UnixSocket.readFully(fd, $0.baseAddress!, 4) }) {
            let n = Int(UInt32(head[0]) | UInt32(head[1]) << 8 | UInt32(head[2]) << 16 | UInt32(head[3]) << 24)
            guard n >= 1, n <= Wire.maxBody else {
                Log.info("session \(id): bad message length \(n)")
                break
            }
            if body.count < n { body = [UInt8](repeating: 0, count: n) }
            guard body.withUnsafeMutableBytes({ UnixSocket.readFully(fd, $0.baseAddress!, n) }) else { break }
            switch body[0] {
            case Wire.msgJSON:
                server.command(Data(body[1..<n]), from: self)
            case Wire.msgInput:
                body.withUnsafeBytes { server.input(UnsafeRawBufferPointer(rebasing: $0[1..<n])) }
            default:
                Log.info("session \(id): unknown message type \(body[0])")
            }
        }
        cond.lock()
        closing = true
        cond.signal()
        cond.unlock()
        writerDone.wait()
        cond.lock()
        fdClosed = true
        close(fd)
        cond.unlock()
        server.ended(self)
    }

    private func writeLoop() {
        let header = UnsafeMutableRawPointer.allocate(byteCount: 4 + Wire.frameHeader, alignment: 8)
        defer { header.deallocate() }
        outer: while true {
            cond.lock()
            while !closing && outbox.isEmpty && !server.frames.hasPending { cond.wait() }
            if closing {
                cond.unlock()
                break
            }
            let batch = outbox
            outbox.removeAll(keepingCapacity: true)
            cond.unlock()
            for msg in batch {
                guard msg.withUnsafeBytes({ UnixSocket.writeFully(fd, $0.baseAddress!, $0.count) }) else { break outer }
            }
            if let next = server.frames.take() {
                let (buf, dropped) = next
                Wire.writeFrameHeader(buf, dropped: dropped, sendNanos: HostClock.nowNanos(), into: header)
                let ok = UnixSocket.writeFully(fd, header, 4 + Wire.frameHeader)
                    && (buf.length == 0 || UnixSocket.writeFully(fd, buf.bytes, buf.length))
                server.frames.recycle(buf)
                guard ok else { break outer }
            }
        }
        cond.lock()
        if !fdClosed { Darwin.shutdown(fd, SHUT_RDWR) }
        cond.unlock()
        writerDone.signal()
    }
}

enum UnixSocket {
    static func address(_ path: String) throws -> sockaddr_un {
        var addr = sockaddr_un()
        let bytes = Array(path.utf8)
        let capacity = MemoryLayout.size(ofValue: addr.sun_path)
        guard bytes.count < capacity else { throw CommandError("socket path is \(bytes.count) bytes, the limit is \(capacity - 1): \(path)") }
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: bytes)
            raw[bytes.count] = 0
        }
        return addr
    }

    private static func withAddress<T>(_ addr: inout sockaddr_un, _ body: (UnsafePointer<sockaddr>, socklen_t) -> T) -> T {
        withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { body($0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
    }

    static func serve(_ path: String) throws -> Int32 {
        var addr = try address(path)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw CommandError("socket(): \(String(cString: strerror(errno)))") }
        guard withAddress(&addr, { bind(fd, $0, $1) }) == 0 else {
            let message = String(cString: strerror(errno))
            close(fd)
            throw CommandError("bind \(path): \(message)")
        }
        chmod(path, 0o600)
        guard listen(fd, 4) == 0 else {
            let message = String(cString: strerror(errno))
            close(fd)
            throw CommandError("listen \(path): \(message)")
        }
        return fd
    }

    static func canConnect(_ path: String) -> Bool {
        guard var addr = try? address(path) else { return false }
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        return withAddress(&addr) { connect(fd, $0, $1) } == 0
    }

    static func configure(_ fd: Int32) {
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        // A Unix stream socket buffers up to the sender's SO_SNDBUF (default 8 KB). 1 MB holds about three
        // 640x480 luma frames: enough to write a frame in one go, small enough that a stalled reader
        // finds only a few stale frames queued (newer ones wait in the FrameStore, replacing each other).
        var size: Int32 = 1 << 20
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, socklen_t(MemoryLayout<Int32>.size))
    }

    static func readFully(_ fd: Int32, _ p: UnsafeMutableRawPointer, _ n: Int) -> Bool {
        var done = 0
        while done < n {
            let k = read(fd, p + done, n - done)
            if k > 0 {
                done += k
            } else if k < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return true
    }

    static func writeFully(_ fd: Int32, _ p: UnsafeRawPointer, _ n: Int) -> Bool {
        var done = 0
        while done < n {
            let k = write(fd, p + done, n - done)
            if k > 0 {
                done += k
            } else if k < 0 && errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return true
    }
}
