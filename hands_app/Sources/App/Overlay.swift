import AppKit
import Foundation

/// What the synthetic keyboard is doing, for the overlay: keys held now and presses so far.
/// Written by CGPoster on the input thread, read on the main thread.
final class KeyLight: @unchecked Sendable {
    private let lock = UnfairLock()
    private var held = Set<UInt16>()  // lock
    private var presses: [UInt16: Int] = [:]  // lock

    func set(_ key: UInt16, down: Bool) {
        lock.with {
            if down {
                if held.insert(key).inserted { presses[key, default: 0] += 1 }
            } else {
                held.remove(key)
            }
        }
    }

    func snapshot() -> (held: Set<UInt16>, presses: [UInt16: Int]) { lock.with { (held, presses) } }
}

/// A cursor people can see. osu! hides the system cursor and the model's skin draws none, so the
/// play looks like it runs by itself; this click-through window above osu! draws the real cursor
/// position with a trail, a ripple on every Z/X press and a Z/X key display beside the window.
/// Capture excludes OsuHands' own windows (ScreenCapture.startNow), so the model never sees it.
@MainActor
final class CursorOverlay: NSObject {
    private let window: NSWindow
    private let view: OverlayView
    private let gate: FrontmostTracker
    private let keys: KeyLight
    private let client: () -> CGRect?  // osu!'s client area, global points (y down), from the clamp
    private var link: CADisplayLink?
    private var lastPresses: [UInt16: Int] = [:]

    var enabled = true

    init(gate: FrontmostTracker, keys: KeyLight, client: @escaping () -> CGRect?) {
        self.gate = gate
        self.keys = keys
        self.client = client
        let frame = NSScreen.screens.first?.frame ?? .zero  // the display with the menu bar: where osu! is captured
        window = NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
        view = OverlayView(frame: NSRect(origin: .zero, size: frame.size))
        super.init()
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.level = .statusBar
        window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenAuxiliary]
        window.isReleasedWhenClosed = false
        window.contentView = view
        // Always ordered in (drawing nothing while hidden), so capture can find and exclude it.
        window.orderFrontRegardless()
        let link = view.displayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        self.link = link
    }

    @objc private func tick(_ link: CADisplayLink) {
        let now = CACurrentMediaTime()
        guard enabled && gate.allowed() else {
            if view.active {
                view.active = false
                view.reset()
                view.needsDisplay = true
            }
            return
        }
        view.active = true
        let snap = keys.snapshot()
        let cursor = window.convertPoint(fromScreen: NSEvent.mouseLocation)
        for (key, n) in snap.presses where n > lastPresses[key, default: 0] {
            view.ripples.append((cursor, now))
        }
        lastPresses = snap.presses
        view.held = snap.held
        view.presses = snap.presses
        view.trail.append((cursor, now))
        view.trail.removeAll { now - $0.1 > OverlayView.trailSeconds }
        view.ripples.removeAll { now - $0.1 > OverlayView.rippleSeconds }
        view.now = now
        if let r = client(), let screen = NSScreen.screens.first {
            // global points, y down -> Cocoa screen coordinates (y up from the bottom of the main display)
            let cocoa = NSRect(x: r.minX, y: screen.frame.maxY - r.maxY, width: r.width, height: r.height)
            view.client = window.convertFromScreen(cocoa)
        } else {
            view.client = nil
        }
        view.needsDisplay = true
    }
}

final class OverlayView: NSView {
    static let trailSeconds = 0.12
    static let rippleSeconds = 0.3
    static let keyNames: [(UInt16, String)] = [(6, "Z"), (7, "X")]

    var active = false
    var trail: [(NSPoint, CFTimeInterval)] = []
    var ripples: [(NSPoint, CFTimeInterval)] = []
    var held = Set<UInt16>()
    var presses: [UInt16: Int] = [:]
    var client: NSRect?
    var now: CFTimeInterval = 0

    private let pink = NSColor(srgbRed: 1.0, green: 0.4, blue: 0.67, alpha: 1)
    private let gold = NSColor(srgbRed: 1.0, green: 0.82, blue: 0.3, alpha: 1)

    func reset() {
        trail.removeAll()
        ripples.removeAll()
    }

    override var isOpaque: Bool { false }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill(using: .copy)
        guard active, let cursor = trail.last?.0 else { return }
        let pressed = !held.isEmpty

        // trail: older segments thinner and fainter
        for i in 1..<max(trail.count, 1) {
            let age = (now - trail[i].1) / Self.trailSeconds
            let path = NSBezierPath()
            path.move(to: trail[i - 1].0)
            path.line(to: trail[i].0)
            path.lineWidth = CGFloat(10 * (1 - age))
            path.lineCapStyle = .round
            pink.withAlphaComponent(CGFloat(0.55 * (1 - age))).setStroke()
            path.stroke()
        }

        // a ring that grows and fades from every key press
        for (p, t) in ripples {
            let k = (now - t) / Self.rippleSeconds
            let r = CGFloat(18 + 34 * k)
            let ring = NSBezierPath(ovalIn: NSRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
            ring.lineWidth = 3
            gold.withAlphaComponent(CGFloat(0.9 * (1 - k))).setStroke()
            ring.stroke()
        }

        // the cursor: bigger and gold while a key is held, like osu!'s cursor expanding on click
        let r: CGFloat = pressed ? 17 : 13
        let disc = NSBezierPath(ovalIn: NSRect(x: cursor.x - r, y: cursor.y - r, width: 2 * r, height: 2 * r))
        (pressed ? gold : pink).withAlphaComponent(0.85).setFill()
        disc.fill()
        disc.lineWidth = 3
        NSColor.white.setStroke()
        disc.stroke()
        let dot = NSBezierPath(ovalIn: NSRect(x: cursor.x - 3, y: cursor.y - 3, width: 6, height: 6))
        NSColor.white.setFill()
        dot.fill()

        // Z/X keys just right of osu!'s window, lit while held, with press counts
        guard let c = client else { return }
        let size: CGFloat = 46
        for (i, (key, name)) in Self.keyNames.enumerated() {
            let box = NSRect(x: c.maxX + 14, y: c.midY + 6 - CGFloat(i) * (size + 12), width: size, height: size)
            let shape = NSBezierPath(roundedRect: box, xRadius: 8, yRadius: 8)
            (held.contains(key) ? gold : NSColor(white: 0.12, alpha: 0.85)).setFill()
            shape.fill()
            shape.lineWidth = 2
            NSColor.white.withAlphaComponent(0.8).setStroke()
            shape.stroke()
            let label = NSAttributedString(string: name, attributes: [
                .font: NSFont.systemFont(ofSize: 18, weight: .bold),
                .foregroundColor: held.contains(key) ? NSColor.black : NSColor.white,
            ])
            let ls = label.size()
            label.draw(at: NSPoint(x: box.midX - ls.width / 2, y: box.midY - ls.height / 2 + 5))
            let count = NSAttributedString(string: "\(presses[key, default: 0])", attributes: [
                .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .medium),
                .foregroundColor: held.contains(key) ? NSColor.black : NSColor(white: 0.8, alpha: 1),
            ])
            let cs = count.size()
            count.draw(at: NSPoint(x: box.midX - cs.width / 2, y: box.minY + 5))
        }
    }
}
