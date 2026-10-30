import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

/// Tracks the frontmost app. Input is only allowed while it is the target (osu!).
///
/// Updated from NSWorkspace activation notifications on the main thread and read from the
/// input thread through a lock, so the per-event check costs no IPC.
final class FrontmostTracker: @unchecked Sendable {
    private let lock = UnfairLock()
    private var bundle = ""  // lock
    private var name = ""  // lock
    private var pid: pid_t = 0  // lock
    private var targetBundle: String  // lock
    private var observer: NSObjectProtocol?

    /// Called with the new `info()` after every change: on the main thread for focus changes,
    /// on the caller's thread when the target changes.
    var onChange: (([String: Any]) -> Void)?

    @MainActor
    init(target: String) {
        targetBundle = target
        set(NSWorkspace.shared.frontmostApplication)
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            self?.set(note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication)
        }
    }

    var target: String {
        get { lock.with { targetBundle } }
        set {
            lock.with { targetBundle = newValue }
            onChange?(info())
        }
    }

    func allowed() -> Bool { lock.with { !targetBundle.isEmpty && bundle == targetBundle } }

    func info() -> [String: Any] {
        lock.with {
            ["bundle_id": bundle, "name": name, "pid": Int(pid), "target_bundle": targetBundle,
             "allowed": !targetBundle.isEmpty && bundle == targetBundle]
        }
    }

    private func set(_ app: NSRunningApplication?) {
        let b = app?.bundleIdentifier ?? "", n = app?.localizedName ?? "", p = app?.processIdentifier ?? 0
        let changed = lock.with { () -> Bool in
            guard b != bundle || p != pid else { return false }
            (bundle, name, pid) = (b, n, p)
            return true
        }
        if changed { onChange?(info()) }
    }
}

/// Posts synthetic input as HID-level CGEvents (the cursor really moves; keys go to the key window).
final class CGPoster: InputPoster {
    static let userData: Int64 = 0x6F73_754A_6576  // "osuJev": tags our events for any event tap
    private let gate: FrontmostTracker
    private let keys: KeyLight
    private let source: CGEventSource?

    init(gate: FrontmostTracker, keys: KeyLight) {
        self.gate = gate
        self.keys = keys
        source = CGEventSource(stateID: .hidSystemState)
        // Never mute the real mouse and keyboard after a synthetic event (the default is 0.25 s),
        // so the user can always take over, e.g. Cmd-Tab away, which also stops our input.
        source?.localEventsSuppressionInterval = 0
        source?.setLocalEventsFilterDuringSuppressionState(
            [.permitLocalMouseEvents, .permitLocalKeyboardEvents, .permitSystemDefinedEvents],
            state: .eventSuppressionStateSuppressionInterval)
        source?.setLocalEventsFilterDuringSuppressionState(
            [.permitLocalMouseEvents, .permitLocalKeyboardEvents, .permitSystemDefinedEvents],
            state: .eventSuppressionStateRemoteMouseDrag)
        source?.userData = Self.userData
    }

    func inputAllowed() -> Bool { gate.allowed() }

    func postMove(x: Double, y: Double, dx: Double, dy: Double) {
        guard let event = CGEvent(mouseEventSource: source, mouseType: .mouseMoved,
                                  mouseCursorPosition: CGPoint(x: x, y: y), mouseButton: .left) else { return }
        // Deltas matter only if the game reads relative motion ("high precision mouse").
        event.setIntegerValueField(.mouseEventDeltaX, value: Int64(dx.rounded()))
        event.setIntegerValueField(.mouseEventDeltaY, value: Int64(dy.rounded()))
        event.post(tap: .cghidEventTap)
    }

    func postMouse(_ button: UInt16, down: Bool, x: Double, y: Double) {
        let type: CGEventType = button == 0 ? (down ? .leftMouseDown : .leftMouseUp) : (down ? .rightMouseDown : .rightMouseUp)
        guard let event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: CGPoint(x: x, y: y),
                                  mouseButton: button == 0 ? .left : .right) else { return }
        event.setIntegerValueField(.mouseEventClickState, value: 1)
        event.flags = []
        event.post(tap: .cghidEventTap)
    }

    /// Modifier keys we hold (ctrl, shift, alt, cmd, left or right); only these flags go on our events,
    /// so nothing leaks in from the real keyboard.
    private static let modifierMasks: [UInt16: CGEventFlags] = [
        59: .maskControl, 62: .maskControl, 56: .maskShift, 60: .maskShift,
        58: .maskAlternate, 61: .maskAlternate, 55: .maskCommand, 54: .maskCommand,
    ]
    private var modifiers: CGEventFlags = []

    func postKey(_ key: UInt16, down: Bool) {
        guard let event = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(key), keyDown: down) else { return }
        if let mask = Self.modifierMasks[key] {
            if down { modifiers.insert(mask) } else { modifiers.remove(mask) }
            event.type = .flagsChanged  // how macOS reports a modifier key (SDL turns it into LCTRL etc.)
        }
        event.flags = modifiers
        event.post(tap: .cghidEventTap)
        if key == 6 || key == 7 { keys.set(key, down: down) }  // Z/X: what the cursor overlay shows
    }
}

enum Permissions {
    static func current() -> [String: Any] {
        [
            "screen_capture": CGPreflightScreenCaptureAccess(),
            "accessibility": AXIsProcessTrusted(),
            "post_event": CGPreflightPostEventAccess(),
        ]
    }

    /// Shows the system prompts (each only once per app identity; after that the user must use
    /// System Settings). Run on the main thread.
    @MainActor
    static func request(screen: Bool, accessibility: Bool, postEvent: Bool) {
        if screen { _ = CGRequestScreenCaptureAccess() }
        if accessibility {
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(options)
        }
        if postEvent { _ = CGRequestPostEventAccess() }
    }

    static let settingsURLs = [
        "screen": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
        "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ]
}
