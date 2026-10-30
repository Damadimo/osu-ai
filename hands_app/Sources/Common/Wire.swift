import Foundation

// OsuHands wire protocol, version 1.
//
// Unix stream socket, default <project>/run/hands.sock (mode 0600). One client at a time:
// a new connection replaces the old one. All integers little-endian. All times are
// nanoseconds of mach_absolute_time, i.e. Python's time.perf_counter_ns() on macOS.
// All geometry is in global display points (origin at the main display's top-left,
// y down), the space CGEvent and CGWindow use.
//
// Framing, both directions:   u32 n | body[n]     body[0] is the message type.
//
// 0x01 JSON, both ways:       0x01 | UTF-8 JSON object
//   request  {"id": int, "cmd": str, ...args}
//   reply    {"id": same, "ok": true, ...} or {"id": same, "ok": false, "error": str}
//   event    {"event": "hello" | "frontmost" | "capture_stopped", ...}   server to client, no id
//
// 0x02 INPUT, client to server:   0x02 | N x 32-byte records
//   u8 kind | u8 0 | u16 key | u32 tag | i64 at_ns | f64 x | f64 y
//   kind 1 MOVE       cursor to (x, y)
//        2 KEY_DOWN   macOS virtual keycode `key` (Z = 6, X = 7)
//        3 KEY_UP
//        4 CANCEL     drop pending actions with this tag (0xFFFFFFFF: all); applied on receipt
//        5 RELEASE    key-up for every key and mouse button the helper is holding
//        6 MOUSE_DOWN mouse button `key` (0 left, 1 right) down at (x, y)
//        7 MOUSE_UP   that button up at (x, y)
//   at_ns <= 0 runs as soon as possible, otherwise at that time on the real-time input thread.
//   Equal times run in arrival order; more than 60 s ahead is rejected.
//   MOVE, KEY_DOWN and MOUSE_DOWN are dropped unless the target app (osu!) is frontmost. KEY_UP or
//   MOUSE_UP of a held key or button and RELEASE always run. Losing focus or disconnecting cancels everything pending and
//   releases held keys; disconnecting also stops the capture.
//
// 0x03 FRAME, server to client:   56-byte header | pixels
//   u8 0x03 | u8 format (1 luma, 4 BGRA) | u16 flags (1 = idle: screen unchanged, no pixels)
//   | u32 config_id | u64 seq | i64 pts_ns | i64 display_ns | i64 recv_ns | i64 send_ns
//   | u16 width | u16 height | u32 dropped | width * height * bytes-per-pixel, rows packed, top first
//   Latest-frame semantics: a frame still waiting to be sent is replaced by a newer one
//   (seq skips; `dropped` counts them).
//
// JSON commands (args in braces; see Server.run):
//   ping, status, stats {reset}, config {target_bundle, clamp}, release, quit,
//   request_permissions {screen, accessibility, post_event}, open_settings {pane},
//   displays, windows {source: "sck" | "cg", all},
//   capture_start {rect, size, fps, format, display, window, queue_depth, resolution, color_space, matrix},
//   capture_update {any capture_start args, merged into the running config}, capture_stop
enum Wire {
    static let version = 1

    static let msgJSON: UInt8 = 0x01
    static let msgInput: UInt8 = 0x02
    static let msgFrame: UInt8 = 0x03

    static let frameHeader = 56  // frame body bytes before the pixels, type byte included
    static let record = 32
    static let maxBody = 1 << 20  // client messages larger than this close the connection

    static let move: UInt8 = 1
    static let keyDown: UInt8 = 2
    static let keyUp: UInt8 = 3
    static let cancel: UInt8 = 4
    static let releaseAll: UInt8 = 5
    static let mouseDown: UInt8 = 6
    static let mouseUp: UInt8 = 7
    static let allTags: UInt32 = 0xFFFF_FFFF

    static let luma: UInt8 = 1
    static let bgra: UInt8 = 4
    static let flagIdle: UInt16 = 1

    /// A complete JSON message (length prefix included), or nil if `obj` is not valid JSON.
    static func encodeJSON(_ obj: [String: Any]) -> Data? {
        guard JSONSerialization.isValidJSONObject(obj),
              let payload = try? JSONSerialization.data(withJSONObject: obj) else { return nil }
        var out = Data(capacity: payload.count + 5)
        withUnsafeBytes(of: UInt32(payload.count + 1).littleEndian) { out.append(contentsOf: $0) }
        out.append(msgJSON)
        out.append(payload)
        return out
    }

    /// Writes the 4-byte length and the 56-byte frame header (60 bytes) at `p`.
    static func writeFrameHeader(_ b: FrameBuffer, dropped: UInt32, sendNanos: Int64, into p: UnsafeMutableRawPointer) {
        p.storeBytes(of: UInt32(frameHeader + b.length).littleEndian, toByteOffset: 0, as: UInt32.self)
        p.storeBytes(of: msgFrame, toByteOffset: 4, as: UInt8.self)
        p.storeBytes(of: b.format, toByteOffset: 5, as: UInt8.self)
        p.storeBytes(of: b.flags.littleEndian, toByteOffset: 6, as: UInt16.self)
        p.storeBytes(of: b.config.littleEndian, toByteOffset: 8, as: UInt32.self)
        p.storeBytes(of: b.seq.littleEndian, toByteOffset: 12, as: UInt64.self)
        p.storeBytes(of: b.ptsNanos.littleEndian, toByteOffset: 20, as: Int64.self)
        p.storeBytes(of: b.displayNanos.littleEndian, toByteOffset: 28, as: Int64.self)
        p.storeBytes(of: b.recvNanos.littleEndian, toByteOffset: 36, as: Int64.self)
        p.storeBytes(of: sendNanos.littleEndian, toByteOffset: 44, as: Int64.self)
        p.storeBytes(of: UInt16(clamping: b.width).littleEndian, toByteOffset: 52, as: UInt16.self)
        p.storeBytes(of: UInt16(clamping: b.height).littleEndian, toByteOffset: 54, as: UInt16.self)
        p.storeBytes(of: dropped.littleEndian, toByteOffset: 56, as: UInt32.self)
    }
}

/// One 32-byte INPUT record.
struct InputRecord {
    var kind: UInt8
    var key: UInt16
    var tag: UInt32
    var atNanos: Int64
    var x: Double
    var y: Double

    /// Records in `raw` (a multiple of 32 bytes).
    static func parse(_ raw: UnsafeRawBufferPointer) -> [InputRecord] {
        stride(from: 0, to: raw.count - Wire.record + 1, by: Wire.record).map { o in
            InputRecord(
                kind: raw.load(fromByteOffset: o, as: UInt8.self),
                key: UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: o + 2, as: UInt16.self)),
                tag: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o + 4, as: UInt32.self)),
                atNanos: Int64(littleEndian: raw.loadUnaligned(fromByteOffset: o + 8, as: Int64.self)),
                x: Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: o + 16, as: UInt64.self))),
                y: Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: o + 24, as: UInt64.self))))
        }
    }
}
