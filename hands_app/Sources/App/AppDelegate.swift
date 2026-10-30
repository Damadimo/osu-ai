import AppKit
import Foundation

/// Menu-bar-only app: runs the socket server and shows its state under a hand icon.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private let socketPath: String
    private let target: String
    private var server: Server?
    private var backend: RealBackend?
    private var overlay: CursorOverlay?
    private var statusItem: NSStatusItem?
    private var activity: NSObjectProtocol?
    private var signalSources: [DispatchSourceSignal] = []

    init(socketPath: String, target: String) {
        self.socketPath = socketPath
        self.target = target
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        Log.info("OsuHands \(version) pid \(getpid()) at \(Bundle.main.bundlePath), target \(target)")
        // No App Nap and no timer coalescing: input timing is the whole point. Idle sleep stays allowed.
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiatedAllowingIdleSystemSleep, .latencyCritical],
            reason: "OsuHands schedules synthetic input to the millisecond")
        let frames = FrameStore()
        let backend = RealBackend(frames: frames, target: target)
        let server = Server(path: socketPath, backend: backend, frames: frames, poster: backend.poster)
        do {
            try server.start()
        } catch {
            Log.info("cannot start: \(error)")
            NSApp.terminate(nil)
            return
        }
        self.backend = backend
        self.server = server
        let overlay = CursorOverlay(gate: backend.tracker, keys: backend.keys) { [weak server] in server?.scheduler.clampRect }
        backend.overlay = overlay
        self.overlay = overlay
        Log.info("permissions \(Permissions.current()), realtime input \(server.scheduler.realtime)")
        installStatusItem()
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { NSApp.terminate(nil) }
            source.resume()
            signalSources.append(source)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        server?.shutdown()
        Log.info("quit")
    }

    // MARK: status item

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "hand.raised", accessibilityDescription: "OsuHands")
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        statusItem = item
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        guard let server, let backend else { return }
        let p = Permissions.current()
        let front = backend.frontmost()
        let capture = backend.captureStatus()
        func line(_ text: String) {
            let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        func action(_ title: String, _ selector: Selector) {
            let item = NSMenuItem(title: title, action: selector, keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        line(server.connected ? "Client connected" : "No client")
        line("Screen Recording: \(p["screen_capture"] as? Bool == true ? "allowed" : "not allowed")")
        line("Accessibility: \(p["accessibility"] as? Bool == true ? "allowed" : "not allowed")")
        if capture["running"] as? Bool == true, let c = capture["config"] as? [String: Any],
           let size = c["size"] as? [Int], let fps = c["fps"] as? Double {
            line("Capturing \(size[0])x\(size[1]) at \(Int(fps)) fps")
        } else {
            line("Not capturing")
        }
        let allowed = front["allowed"] as? Bool == true
        line("Input \(allowed ? "live" : "paused"): \(front["name"] as? String ?? "?") is frontmost")
        menu.addItem(.separator())
        action("Request Permissions…", #selector(requestPermissions))
        action("Open Screen Recording Settings…", #selector(openScreenSettings))
        action("Open Accessibility Settings…", #selector(openAccessibilitySettings))
        menu.addItem(.separator())
        action("Quit OsuHands", #selector(quit))
    }

    @objc private func requestPermissions() {
        Permissions.request(screen: true, accessibility: true, postEvent: false)
    }

    @objc private func openScreenSettings() { open(Permissions.settingsURLs["screen"]!) }

    @objc private func openAccessibilitySettings() { open(Permissions.settingsURLs["accessibility"]!) }

    private func open(_ string: String) {
        if let url = URL(string: string) { NSWorkspace.shared.open(url) }
    }

    @objc private func quit() { NSApp.terminate(nil) }
}
