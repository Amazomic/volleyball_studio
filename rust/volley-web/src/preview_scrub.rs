//! Pure, deterministic keyboard-scrub policy for the browser's video player.
//! DOM events and HTMLVideoElement.currentTime stay in the JS platform adapter.

const HOLD_DELAY_MS: f64 = 180.0;
const HOLD_DOUBLING_MS: f64 = 400.0;
const INITIAL_SECONDS_PER_SECOND: f64 = 1.0;
const MAX_SECONDS_PER_SECOND: f64 = 80.0;
const MAX_FRAME_MS: f64 = 200.0;

fn signed_direction(direction: i32) -> f64 {
    match direction {
        -1 => -1.0,
        1 => 1.0,
        _ => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn volley_preview_seek_step(
    position: f64,
    direction: i32,
    duration: f64,
    steps: u32,
) -> f64 {
    if !position.is_finite()
        || !duration.is_finite()
        || duration <= 0.0
        || signed_direction(direction) == 0.0
        || steps == 0
    {
        return position;
    }
    let target = (position * 1_000_000.0).round()
        + signed_direction(direction) * f64::from(steps) * 100_000.0;
    (target / 1_000_000.0).clamp(0.0, (duration - 1e-7).max(0.0))
}

/// Source seconds to advance for this rendered frame while holding an arrow.
/// Returns zero during the initial grace period. Frame latency and velocity are
/// bounded, so a suspended browser tab cannot unexpectedly jump many minutes.
#[no_mangle]
pub extern "C" fn volley_preview_seek_hold_delta(
    direction: i32,
    held_ms: f64,
    frame_ms: f64,
) -> f64 {
    let signed = signed_direction(direction);
    if signed == 0.0
        || !held_ms.is_finite()
        || !frame_ms.is_finite()
        || held_ms < HOLD_DELAY_MS
        || frame_ms <= 0.0
    {
        return 0.0;
    }
    let doublings = (held_ms - HOLD_DELAY_MS) / HOLD_DOUBLING_MS;
    let velocity =
        (INITIAL_SECONDS_PER_SECOND * 2.0_f64.powf(doublings)).min(MAX_SECONDS_PER_SECOND);
    signed * velocity * frame_ms.min(MAX_FRAME_MS) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_advances_exactly_one_tenth_of_a_second() {
        for start in [0.0, 0.017, 4.123, 1234.567] {
            let next = volley_preview_seek_step(start, 1, 2000.0, 1);
            assert!((next - start - 0.1).abs() < 1e-10);
            assert!((volley_preview_seek_step(next, -1, 2000.0, 1) - start).abs() < 1e-10);
            let mut time = start;
            for _ in 0..10 {
                time = volley_preview_seek_step(time, 1, 2000.0, 1);
            }
            assert!((time - start - 1.0).abs() < 1e-10);
        }
        assert_eq!(volley_preview_seek_step(0.0, -1, 60.0, 1), 0.0);
        assert!(volley_preview_seek_step(60.0, 1, 60.0, 1) < 60.0);
        assert_eq!(volley_preview_seek_step(4.0, 1, 60.0, 10), 5.0);
        assert_eq!(volley_preview_seek_step(0.7, 1, 60.0, 1), 0.8);
        assert_eq!(volley_preview_seek_step(4.0, 0, 60.0, 1), 4.0);
        assert_eq!(volley_preview_seek_step(4.0, 1, 60.0, 0), 4.0);
    }

    #[test]
    fn hold_starts_only_after_grace_period() {
        for held in [0.0, 16.0, 179.99] {
            assert_eq!(volley_preview_seek_hold_delta(1, held, 100.0), 0.0);
        }
        assert_eq!(volley_preview_seek_hold_delta(1, 180.0, 100.0), 0.1);
        assert_eq!(volley_preview_seek_hold_delta(-1, 180.0, 100.0), -0.1);
    }

    #[test]
    fn holding_accelerates_exponentially_and_stops_at_limit() {
        let t0 = volley_preview_seek_hold_delta(1, 180.0, 100.0);
        let t1 = volley_preview_seek_hold_delta(1, 580.0, 100.0);
        let t2 = volley_preview_seek_hold_delta(1, 980.0, 100.0);
        assert!((t1 - 2.0 * t0).abs() < 1e-10);
        assert!((t2 - 4.0 * t0).abs() < 1e-10);
        assert_eq!(volley_preview_seek_hold_delta(1, 30_000.0, 100.0), 8.0);
        assert_eq!(volley_preview_seek_hold_delta(1, 30_000.0, 10_000.0), 16.0);
    }

    #[test]
    fn invalid_inputs_and_background_frame_stalls_are_safe() {
        for held in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(volley_preview_seek_hold_delta(1, held, 100.0), 0.0);
        }
        for frame in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(volley_preview_seek_hold_delta(1, 500.0, frame), 0.0);
        }
        assert_eq!(volley_preview_seek_hold_delta(2, 500.0, 100.0), 0.0);
        assert_eq!(volley_preview_seek_hold_delta(-1, 30_000.0, 500.0), -16.0);
    }
}
