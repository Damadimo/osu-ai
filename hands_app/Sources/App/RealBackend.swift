import AppKit
import CoreGraphics
import Foundation
import ScreenCaptureKit

/// The real backend: ScreenCaptureKit frames, CGEvent input, TCC permissions, NSWorkspace focus.
final class RealBackend: Backend {
    let name = "screencapturekit"
    let capture: ScreenCapture
    let tracker: FrontmostTracker
    let poster: CGPoster
    let keys = KeyLight()
    var overlay: CursorOverlay?  // set on the main thread by AppDelegate
    var onEvent: (([String: Any]) -> Void)?

    @MainActor
    init(frames: FrameStore, target: String) {
        capture = ScreenCapture(frames: frames)
        tracker = FrontmostTracker(target: target)
        poster = CGPoster(gate: tracker, keys: keys)
        capture.onStopped = { [weak self] reason in self?.onEvent?(["event": "capture_stopped", "error": reason]) }
        tracker.onChange = { [weak self] info in
            var e = info
            e["event"] = "frontmost"
            self?.onEvent?(e)
        }
    }

    /// The bundle id of /Applications/osu!.app, or osu!lazer's known id if it is not installed.
    static func installedOsuBundleID() -> String {
        let plist = URL(fileURLWithPath: "/Applications/osu!.app/Contents/Info.plist")
        return (NSDictionary(contentsOf: plist)?["CFBundleIdentifier"] as? String) ?? "sh.ppy.osu.lazer"
    }

    var targetBundle: String {
        get { tracker.target }
        set { tracker.target = newValue }
    }

    var captureConfig: CaptureConfig? { capture.config }

    func permissions() -> [String: Any] { Permissions.current() }

    func frontmost() -> [String: Any] { tracker.info() }

    func captureStatus() -> [String: Any] { capture.describe() }

    func requestPermissions(_ args: [String: Any], reply: @escaping Reply) {
        let screen = args["screen"] as? Bool ?? true
        let accessibility = args["accessibility"] as? Bool ?? true
        let postEvent = args["post_event"] as? Bool ?? false
        DispatchQueue.main.async {
            Permissions.request(screen: screen, accessibility: accessibility, postEvent: postEvent)
            reply(.success(Permissions.current()))
        }
    }

    func displays(reply: @escaping Reply) {
        DispatchQueue.main.async {
            var ids = [CGDirectDisplayID](repeating: 0, count: 16)
            var count: UInt32 = 0
            CGGetActiveDisplayList(UInt32(ids.count), &ids, &count)
            let list = ids.prefix(Int(count)).map { id -> [String: Any] in
                var d: [String: Any] = ["id": Int(id), "main": CGDisplayIsMain(id) != 0, "frame": rectArray(CGDisplayBounds(id))]
                if let mode = CGDisplayCopyDisplayMode(id) {
                    d["pixels"] = [mode.pixelWidth, mode.pixelHeight]
                    d["scale"] = Double(mode.pixelWidth) / Double(max(mode.width, 1))
                    d["refresh_hz"] = mode.refreshRate
                }
                let key = NSDeviceDescriptionKey("NSScreenNumber")
                if let screen = NSScreen.screens.first(where: { ($0.deviceDescription[key] as? NSNumber)?.uint32Value == id }) {
                    d["name"] = screen.localizedName
                    d["backing_scale"] = Double(screen.backingScaleFactor)
                    d["max_fps"] = screen.maximumFramesPerSecond
                    let inset = screen.safeAreaInsets  // the notch: top inset on built-in MacBook displays
                    d["safe_area_insets"] = ["top": Double(inset.top), "left": Double(inset.left),
                                             "bottom": Double(inset.bottom), "right": Double(inset.right)]
                }
                return d
            }
            reply(.success(["displays": list]))
        }
    }

    func windows(_ args: [String: Any], reply: @escaping Reply) {
        let all = args["all"] as? Bool ?? false
        if args["source"] as? String == "cg" {
            DispatchQueue.main.async { reply(.success(["windows": Self.cgWindows(all: all), "source": "cg"])) }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            switch ScreenCapture.shareableContent(onScreenOnly: !all) {
            case .failure(let e):
                reply(.failure(e))
            case .success(let content):
                let list = content.windows.map { w -> [String: Any] in
                    [
                        "id": Int(w.windowID), "title": w.title ?? "", "app": w.owningApplication?.applicationName ?? "",
                        "bundle_id": w.owningApplication?.bundleIdentifier ?? "",
                        "pid": Int(w.owningApplication?.processID ?? 0), "frame": rectArray(w.frame),
                        "layer": w.windowLayer, "on_screen": w.isOnScreen, "active": w.isActive,
                    ]
                }
                reply(.success(["windows": list, "source": "sck"]))
            }
        }
    }

    /// CGWindowList needs no Screen Recording permission for bounds and owners (titles are
    /// empty without it). Front-to-back order.
    private static func cgWindows(all: Bool) -> [[String: Any]] {
        let options: CGWindowListOption = all ? [.optionAll] : [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else { return [] }
        return list.map { d in
            let pid = (d[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value ?? 0
            var frame = CGRect.zero
            if let bounds = d[kCGWindowBounds as String] as? NSDictionary {
                frame = CGRect(dictionaryRepresentation: bounds as CFDictionary) ?? .zero
            }
            return [
                "id": (d[kCGWindowNumber as String] as? NSNumber)?.intValue ?? 0,
                "title": d[kCGWindowName as String] as? String ?? "",
                "app": d[kCGWindowOwnerName as String] as? String ?? "",
                "bundle_id": NSRunningApplication(processIdentifier: pid)?.bundleIdentifier ?? "",
                "pid": Int(pid), "frame": rectArray(frame),
                "layer": (d[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0,
                "on_screen": (d[kCGWindowIsOnscreen as String] as? NSNumber)?.boolValue ?? false,
            ]
        }
    }

    func startCapture(_ config: CaptureConfig, reply: @escaping Reply) { capture.start(config, reply: reply) }

    func updateCapture(_ config: CaptureConfig, reply: @escaping Reply) { capture.update(config, reply: reply) }

    func stopCapture(reply: @escaping Reply) { capture.stop(reply: reply) }

    func extra(_ cmd: String, _ args: [String: Any], reply: @escaping Reply) -> Bool {
        if cmd == "overlay" {  // {"show": bool}: the visible cursor and Z/X keys (never captured)
            let show = args["show"] as? Bool
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if let show { self.overlay?.enabled = show }
                    reply(.success(["show": self.overlay?.enabled ?? false]))
                }
            }
            return true
        }
        guard cmd == "open_settings" else { return false }
        let pane = args["pane"] as? String ?? "screen"
        guard let string = Permissions.settingsURLs[pane], let url = URL(string: string) else {
            reply(.failure(CommandError("pane must be screen or accessibility")))
            return true
        }
        DispatchQueue.main.async {
            NSWorkspace.shared.open(url)
            reply(.success(["opened": string]))
        }
        return true
    }

    func quit() {
        DispatchQueue.main.async { NSApp.terminate(nil) }
    }
}
