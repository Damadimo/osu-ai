import CoreGraphics
import Darwin
import Foundation

/// One timed input action. `deadline` is in mach ticks; 0 runs as soon as possible.
struct InputAction {
    var deadline: UInt64
    var kind: UInt8
    var key: UInt16 = 0
    var tag: UInt32 = 0
    var x: Double = 0
    var y: Double = 0
    var order: UInt64 = 0

    @inline(__always) func before(_ o: InputAction) -> Bool {
        deadline != o.deadline ? deadline < o.deadline : order < o.order
    }
}

/// Where executed actions go: CGEvents in the app, a log in the self-test.
protocol InputPoster: AnyObject {
    /// Whether synthetic input may reach the target app right now (it is frontmost).
    func inputAllowed() -> Bool
    func postMove(x: Double, y: Double, dx: Double, dy: Double)
    func postKey(_ key: UInt16, down: Bool)
    func postMouse(_ button: UInt16, down: Bool, x: Double, y: Double)
}

/// Runs input actions at their deadlines on one real-time thread.
///
/// The thread sleeps on a condition variable until 0.5 ms before the earliest deadline (so a
/// newly submitted, earlier action can wake it), then mach_wait_until()s the exact tick. With
/// the Mach time-constraint policy the wakeup lands within tens of microseconds.
final class InputScheduler {
    static let maxAheadNanos: UInt64 = 60_000_000_000

    private let poster: InputPoster
    private let mutex = UnsafeMutablePointer<pthread_mutex_t>.allocate(capacity: 1)
    private let cond = UnsafeMutablePointer<pthread_cond_t>.allocate(capacity: 1)
    private var heap: [InputAction] = []  // mutex
    private var order: UInt64 = 0  // mutex
    private var stopped = false  // mutex
    private let margin = HostClock.ticks(500_000)

    private let keys = UnfairLock()
    private var held = Set<UInt16>()  // keys
    private var heldButtons = Set<UInt16>()  // keys
    private var closed = false  // keys
    private var clamp: CGRect?  // keys
    private var last: (x: Double, y: Double)?  // input thread only

    private let statsLock = UnfairLock()
    private var stats = Stats()  // statsLock
    private var isRealtime = false  // statsLock

    struct Stats {
        var executed = 0, blocked = 0, rejected = 0, cancelled = 0
        var timed = 0
        var lateSum: UInt64 = 0, lateMax: UInt64 = 0  // ns past the deadline
        var lateHist = [Int](repeating: 0, count: 7)
        static let bounds: [UInt64] = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
        static let labels = ["<50us", "<100us", "<250us", "<500us", "<1ms", "<2ms", ">=2ms"]
    }

    init(poster: InputPoster) {
        self.poster = poster
        pthread_mutex_init(mutex, nil)
        pthread_cond_init(cond, nil)
        let thread = Thread { self.run() }
        thread.name = "com.osujev.hands.input"
        thread.qualityOfService = .userInteractive
        thread.stackSize = 1 << 20
        thread.start()
    }

    var realtime: Bool { statsLock.with { isRealtime } }

    var clampRect: CGRect? {
        get { keys.with { clamp } }
        set { keys.with { clamp = newValue } }
    }

    func submit(_ actions: [InputAction]) {
        guard !actions.isEmpty else { return }
        pthread_mutex_lock(mutex)
        for var a in actions {
            order &+= 1
            a.order = order
            push(a)
        }
        pthread_cond_signal(cond)
        pthread_mutex_unlock(mutex)
    }

    /// Drops pending actions with `tag` (Wire.allTags: all of them). Returns how many.
    @discardableResult
    func cancel(tag: UInt32) -> Int {
        pthread_mutex_lock(mutex)
        let before = heap.count
        if tag == Wire.allTags {
            heap.removeAll()
        } else {
            heap.removeAll { $0.tag == tag }
            heapify()
        }
        let removed = before - heap.count
        pthread_cond_signal(cond)
        pthread_mutex_unlock(mutex)
        statsLock.with { stats.cancelled += removed }
        return removed
    }

    func releaseAll() { submit([InputAction(deadline: 0, kind: Wire.releaseAll)]) }

    /// Cancels everything pending, then releases held keys ahead of anything submitted later.
    func reset() {
        cancel(tag: Wire.allTags)
        releaseAll()
    }

    /// For shutdown: stops the thread and releases held keys from the calling thread.
    func stopAndRelease() {
        pthread_mutex_lock(mutex)
        stopped = true
        heap.removeAll()
        pthread_cond_signal(cond)
        pthread_mutex_unlock(mutex)
        keys.with {
            closed = true
            for k in held { poster.postKey(k, down: false) }
            held.removeAll()
            for b in heldButtons { poster.postMouse(b, down: false, x: last?.x ?? 0, y: last?.y ?? 0) }
            heldButtons.removeAll()
        }
    }

    func noteRejected(_ n: Int) { statsLock.with { stats.rejected += n } }

    func statsJSON(reset: Bool) -> [String: Any] {
        let pending: Int = {
            pthread_mutex_lock(mutex)
            defer { pthread_mutex_unlock(mutex) }
            return heap.count
        }()
        let (held, buttons) = keys.with { (self.held.sorted().map(Int.init), self.heldButtons.sorted().map(Int.init)) }
        return statsLock.with {
            let s = stats
            if reset { stats = Stats() }
            var hist: [String: Int] = [:]
            for (label, n) in zip(Stats.labels, s.lateHist) { hist[label] = n }
            return [
                "executed": s.executed, "blocked": s.blocked, "rejected": s.rejected, "cancelled": s.cancelled,
                "timed": s.timed, "late_mean_us": s.timed > 0 ? Double(s.lateSum) / Double(s.timed) / 1000 : 0,
                "late_max_us": Double(s.lateMax) / 1000, "late_hist": hist,
                "pending": pending, "held_keys": held, "held_buttons": buttons, "realtime": isRealtime,
            ]
        }
    }

    // MARK: input thread

    private func run() {
        let rt = makeRealtime()
        statsLock.with { isRealtime = rt }
        pthread_mutex_lock(mutex)
        while !stopped {
            guard let next = heap.first else {
                pthread_cond_wait(cond, mutex)
                continue
            }
            let now = mach_absolute_time()
            if next.deadline <= now {
                let action = pop()
                pthread_mutex_unlock(mutex)
                execute(action)
                pthread_mutex_lock(mutex)
            } else if next.deadline - now > margin {
                let wait = HostClock.nanos(next.deadline - now - margin)
                var ts = timespec(tv_sec: Int(wait / 1_000_000_000), tv_nsec: Int(wait % 1_000_000_000))
                pthread_cond_timedwait_relative_np(cond, mutex, &ts)
            } else {
                pthread_mutex_unlock(mutex)
                mach_wait_until(next.deadline)
                pthread_mutex_lock(mutex)
            }
        }
        pthread_mutex_unlock(mutex)
    }

    private func execute(_ a: InputAction) {
        let start = mach_absolute_time()
        let allowed = poster.inputAllowed()
        var ran = true
        switch a.kind {
        case Wire.move:
            if allowed {
                let (x, y) = clamped(a.x, a.y)
                let (dx, dy) = last.map { (x - $0.x, y - $0.y) } ?? (0, 0)
                poster.postMove(x: x, y: y, dx: dx, dy: dy)
                last = (x, y)
            } else {
                ran = false
            }
        case Wire.keyDown:
            ran = keys.with {
                guard allowed, !closed, !held.contains(a.key) else { return false }
                held.insert(a.key)
                poster.postKey(a.key, down: true)
                return true
            }
        case Wire.keyUp:
            ran = keys.with {
                guard held.remove(a.key) != nil || (allowed && !closed) else { return false }
                poster.postKey(a.key, down: false)
                return true
            }
        case Wire.mouseDown:
            let (x, y) = clamped(a.x, a.y)
            ran = keys.with {
                guard allowed, !closed, a.key <= 1, !heldButtons.contains(a.key) else { return false }
                heldButtons.insert(a.key)
                poster.postMouse(a.key, down: true, x: x, y: y)
                return true
            }
            if ran { last = (x, y) }
        case Wire.mouseUp:
            let (x, y) = clamped(a.x, a.y)
            ran = keys.with {
                guard heldButtons.remove(a.key) != nil || (allowed && !closed && a.key <= 1) else { return false }
                poster.postMouse(a.key, down: false, x: x, y: y)
                return true
            }
            if ran { last = (x, y) }
        case Wire.releaseAll:
            keys.with {
                for k in held { poster.postKey(k, down: false) }
                held.removeAll()
                for b in heldButtons { poster.postMouse(b, down: false, x: last?.x ?? 0, y: last?.y ?? 0) }
                heldButtons.removeAll()
            }
        default:
            ran = false
        }
        statsLock.with {
            guard ran else {
                stats.blocked += 1
                return
            }
            stats.executed += 1
            guard a.deadline > 0 else { return }
            let late = start > a.deadline ? HostClock.nanos(start - a.deadline) : 0
            stats.timed += 1
            stats.lateSum += late
            stats.lateMax = max(stats.lateMax, late)
            stats.lateHist[Stats.bounds.firstIndex { late < $0 } ?? Stats.bounds.count] += 1
        }
    }

    private func clamped(_ x: Double, _ y: Double) -> (Double, Double) {
        guard let c = clampRect else { return (x, y) }
        return (min(max(x, Double(c.minX)), Double(c.maxX) - 1), min(max(y, Double(c.minY)), Double(c.maxY) - 1))
    }

    /// Mach time-constraint (real-time) scheduling for this thread: aperiodic, 0.5 ms of work per
    /// 1 ms window. Falls back to plain user-interactive QoS if the kernel refuses.
    private func makeRealtime() -> Bool {
        var policy = thread_time_constraint_policy_data_t(
            period: 0,
            computation: UInt32(HostClock.ticks(500_000)),
            constraint: UInt32(HostClock.ticks(1_000_000)),
            preemptible: 1)
        let count = mach_msg_type_number_t(
            MemoryLayout<thread_time_constraint_policy_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &policy) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                thread_policy_set(pthread_mach_thread_np(pthread_self()),
                                  thread_policy_flavor_t(THREAD_TIME_CONSTRAINT_POLICY), $0, count)
            }
        }
        if kr != KERN_SUCCESS { Log.info("input thread: time-constraint policy refused (\(kr)), using QoS only") }
        return kr == KERN_SUCCESS
    }

    // MARK: binary heap ordered by (deadline, arrival)

    private func push(_ a: InputAction) {
        heap.append(a)
        var i = heap.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            guard heap[i].before(heap[parent]) else { break }
            heap.swapAt(i, parent)
            i = parent
        }
    }

    private func pop() -> InputAction {
        let top = heap[0]
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            siftDown(0)
        }
        return top
    }

    private func siftDown(_ start: Int) {
        var i = start
        while true {
            let l = 2 * i + 1, r = l + 1
            var m = i
            if l < heap.count && heap[l].before(heap[m]) { m = l }
            if r < heap.count && heap[r].before(heap[m]) { m = r }
            if m == i { return }
            heap.swapAt(i, m)
            i = m
        }
    }

    private func heapify() {
        guard heap.count > 1 else { return }
        for i in stride(from: heap.count / 2 - 1, through: 0, by: -1) { siftDown(i) }
    }
}
