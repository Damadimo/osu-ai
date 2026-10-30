import Darwin

/// The one clock on the wire: mach_absolute_time() converted to nanoseconds.
///
/// Python's time.perf_counter_ns() on macOS is exactly this value: CPython calls
/// mach_absolute_time() and scales it by mach_timebase_info with truncating integer
/// math (time.get_clock_info("perf_counter").implementation == "mach_absolute_time()").
/// clock_gettime_nsec_np(CLOCK_UPTIME_RAW) is the same clock. It stops while the Mac
/// sleeps. CLOCK_MONOTONIC and CLOCK_MONOTONIC_RAW are different clocks on macOS
/// (they keep counting through sleep), so never mix them in.
enum HostClock {
    static let timebase: (numer: UInt64, denom: UInt64) = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return (UInt64(info.numer), UInt64(info.denom))
    }()

    @inline(__always) static func now() -> UInt64 { mach_absolute_time() }

    /// Ticks to ns, floor(ticks * numer / denom) without overflow, the same rounding as CPython.
    static func nanos(_ ticks: UInt64) -> UInt64 {
        let (n, d) = timebase
        if n == d { return ticks }
        return ticks / d * n + ticks % d * n / d
    }

    /// ns to ticks, rounded up so a deadline never fires early.
    static func ticks(_ nanos: UInt64) -> UInt64 {
        let (n, d) = timebase
        if n == d { return nanos }
        return nanos / n * d + (nanos % n * d + n - 1) / n
    }

    static func nowNanos() -> Int64 { Int64(nanos(now())) }
}
