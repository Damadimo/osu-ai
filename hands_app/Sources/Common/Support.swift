import CoreGraphics
import Foundation
import os

/// os_unfair_lock at a stable heap address (it must never live inline in a Swift object).
final class UnfairLock {
    private let p: UnsafeMutablePointer<os_unfair_lock>

    init() {
        p = .allocate(capacity: 1)
        p.initialize(to: os_unfair_lock())
    }

    deinit {
        p.deinitialize(count: 1)
        p.deallocate()
    }

    @inline(__always) func with<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(p)
        defer { os_unfair_lock_unlock(p) }
        return try body()
    }
}

/// Appends timestamped lines to run/hands.log (stderr until a file is opened).
enum Log {
    private static let queue = DispatchQueue(label: "com.osujev.hands.log")
    nonisolated(unsafe) private static var handle: FileHandle?
    nonisolated(unsafe) private static var formatter: ISO8601DateFormatter?

    static func open(_ url: URL) {
        queue.sync {
            let fm = FileManager.default
            if let size = (try? fm.attributesOfItem(atPath: url.path))?[.size] as? Int, size > 4 << 20 {
                try? fm.removeItem(at: url)
            }
            if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
            handle = try? FileHandle(forWritingTo: url)
            _ = try? handle?.seekToEnd()
        }
    }

    static func info(_ message: String) {
        let date = Date()
        queue.async {
            if formatter == nil {
                let f = ISO8601DateFormatter()
                f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                formatter = f
            }
            let line = Data("\(formatter!.string(from: date)) \(message)\n".utf8)
            if let handle { try? handle.write(contentsOf: line) } else { FileHandle.standardError.write(line) }
        }
    }
}

typealias Reply = (Result<[String: Any], CommandError>) -> Void

struct CommandError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// A JSON array of numbers, or nil.
func numbers(_ value: Any?) -> [Double]? {
    guard let array = value as? [Any] else { return nil }
    let out = array.compactMap { ($0 as? NSNumber)?.doubleValue }
    return out.count == array.count ? out : nil
}

func rectArray(_ r: CGRect) -> [Double] { [Double(r.minX), Double(r.minY), Double(r.width), Double(r.height)] }

/// Blocks the calling (non-main) queue until `body` calls back, or returns nil after `timeout` seconds.
func waitFor<T>(_ timeout: Double, _ body: (@escaping (T) -> Void) -> Void) -> T? {
    let done = DispatchSemaphore(value: 0)
    let lock = UnfairLock()
    var result: T?
    body { value in
        lock.with { if result == nil { result = value } }
        done.signal()
    }
    guard done.wait(timeout: .now() + timeout) == .success else { return nil }
    return lock.with { result }
}

/// An optional error that can be passed through waitFor.
struct Outcome {
    let error: Error?
}
