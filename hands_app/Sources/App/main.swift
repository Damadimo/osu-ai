// OsuHands: screen frames and synthetic input for the osu-ai agent, over a Unix socket.
//
// Launch with `open -g OsuHands.app [--args --socket PATH --target BUNDLE_ID]` so that the
// app is its own responsible process and macOS attributes Screen Recording and Accessibility
// to com.osujev.hands (not to the terminal). Protocol: Sources/Common/Wire.swift.
import AppKit
import Foundation

func argument(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

/// <project>/run/hands.sock when the app sits in <project>/hands_app, else ~/projects/osu-ai/run.
func defaultSocketPath() -> String {
    let parent = URL(fileURLWithPath: Bundle.main.bundlePath).deletingLastPathComponent()
    if parent.lastPathComponent == "hands_app" {
        return parent.deletingLastPathComponent().appendingPathComponent("run/hands.sock").path
    }
    return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("projects/osu-ai/run/hands.sock").path
}

let env = ProcessInfo.processInfo.environment
let socketPath = argument("--socket") ?? env["OSUHANDS_SOCKET"] ?? defaultSocketPath()
let target = argument("--target") ?? env["OSUHANDS_TARGET"] ?? RealBackend.installedOsuBundleID()

signal(SIGPIPE, SIG_IGN)
try? FileManager.default.createDirectory(
    atPath: (socketPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
Log.open(URL(fileURLWithPath: (socketPath as NSString).deletingLastPathComponent).appendingPathComponent("hands.log"))

MainActor.assumeIsolated {  // top-level code runs on the main thread
    let app = NSApplication.shared
    let delegate = AppDelegate(socketPath: socketPath, target: target)
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()  // never returns while the app runs; `delegate` stays alive until then
}
