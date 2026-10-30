import Foundation

/// One captured frame: packed pixel rows plus the FRAME header fields.
final class FrameBuffer {
    private(set) var bytes: UnsafeMutableRawPointer
    private(set) var capacity: Int
    var length = 0
    var format = Wire.luma
    var flags: UInt16 = 0
    var config: UInt32 = 0
    var seq: UInt64 = 0
    var ptsNanos: Int64 = 0
    var displayNanos: Int64 = 0
    var recvNanos: Int64 = 0
    var width = 0
    var height = 0

    init(capacity: Int = 640 * 480) {
        self.capacity = capacity
        bytes = .allocate(byteCount: capacity, alignment: 64)
    }

    deinit { bytes.deallocate() }

    func reserve(_ count: Int) {
        guard count > capacity else { return }
        bytes.deallocate()
        bytes = .allocate(byteCount: count, alignment: 64)
        capacity = count
    }
}

/// Hands frames from the capture queue to the socket writer with latest-frame semantics.
///
/// At most three buffers circulate: one being filled, one pending, one being sent. Publishing
/// while a frame is still pending replaces it (counted in `dropped`), so a slow client always
/// gets the newest frame and capture never blocks on the socket.
final class FrameStore {
    private let lock = UnfairLock()
    private var spare: [FrameBuffer] = []
    private var pending: FrameBuffer?
    private var waker: (() -> Void)?
    private var published = 0
    private var dropped: UInt32 = 0
    private var idleMerged = 0

    func setWaker(_ waker: (() -> Void)?) { lock.with { self.waker = waker } }

    var hasPending: Bool { lock.with { pending != nil } }

    func checkout() -> FrameBuffer { lock.with { spare.popLast() } ?? FrameBuffer() }

    func recycle(_ b: FrameBuffer) { lock.with { if spare.count < 3 { spare.append(b) } } }

    func publish(_ b: FrameBuffer) {
        let wake: (() -> Void)? = lock.with {
            if let old = pending {
                dropped &+= 1
                if spare.count < 3 { spare.append(old) }
            }
            pending = b
            published += 1
            return waker
        }
        wake?()
    }

    /// An idle frame (screen unchanged) carries no pixels. If a full frame is still pending the
    /// client will get those same pixels anyway, so the idle notice is merged into it.
    func publishIdle(seq: UInt64, config: UInt32, format: UInt8, pts: Int64, shown: Int64, recv: Int64) {
        if lock.with({ pending != nil }) {
            lock.with { idleMerged += 1 }
            return
        }
        let b = checkout()
        b.length = 0
        b.width = 0
        b.height = 0
        b.flags = Wire.flagIdle
        b.format = format
        b.seq = seq
        b.config = config
        b.ptsNanos = pts
        b.displayNanos = shown
        b.recvNanos = recv
        publish(b)
    }

    /// The pending frame and the drop count so far, or nil.
    func take() -> (FrameBuffer, UInt32)? {
        lock.with {
            guard let b = pending else { return nil }
            pending = nil
            return (b, dropped)
        }
    }

    func reset() {
        lock.with {
            if let b = pending, spare.count < 3 { spare.append(b) }
            pending = nil
            published = 0
            dropped = 0
            idleMerged = 0
        }
    }

    func statsJSON() -> [String: Any] {
        lock.with { ["published": published, "dropped": Int(dropped), "idle_merged": idleMerged, "pending": pending != nil] }
    }
}
