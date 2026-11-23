use anyhow::{ensure, Result};

/// Windows of simulated time per unit of the wait threshold.
const PERIOD_PER_WAIT: f64 = 4.0;
/// Consecutive windows over the threshold that saturate a run.
const WINDOWS: u32 = 2;

/// BookSim-style stability check. Every `PERIOD_PER_WAIT × wait` seconds of
/// simulated time it takes the mean first-token wait (TTFT minus prefill) of
/// the calls completed in that window; the run is saturated once `WINDOWS`
/// consecutive windows exceed `wait`. Windows without completions neither
/// count nor reset.
pub struct Saturation {
    period: f64,
    wait: f64,
    next: f64,
    last: (u64, f64),
    streak: u32,
}

impl Saturation {
    pub fn new(wait: f64) -> Result<Self> {
        ensure!(
            wait.is_finite() && wait > 0.0,
            "saturation wait must be positive"
        );
        let period = PERIOD_PER_WAIT * wait;
        Ok(Self {
            period,
            wait,
            next: period,
            last: (0, 0.0),
            streak: 0,
        })
    }

    /// Takes the clock and the cumulative (calls, summed wait); returns the
    /// last window's mean wait once the run is saturated.
    pub fn check(&mut self, time: f64, (calls, waits): (u64, f64)) -> Option<f64> {
        if time < self.next {
            return None;
        }
        while self.next <= time {
            self.next += self.period;
        }
        let (count, sum) = (calls - self.last.0, waits - self.last.1);
        self.last = (calls, waits);
        if count == 0 {
            return None;
        }
        let mean = sum / count as f64;
        self.streak = if mean > self.wait { self.streak + 1 } else { 0 };
        (self.streak >= WINDOWS).then_some(mean)
    }
}

#[derive(Debug)]
pub struct Saturated {
    pub time: f64,
    pub mean_wait: f64,
}

impl std::fmt::Display for Saturated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "saturated at {} s: mean first-token wait {} s",
            self.time, self.mean_wait
        )
    }
}

impl std::error::Error for Saturated {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consecutive_slow_windows_saturate_and_a_fast_one_resets() {
        let mut s = Saturation::new(2.5).unwrap(); // 10 s windows
        assert_eq!(s.check(5.0, (1, 9.0)), None); // inside the first window
        assert_eq!(s.check(10.0, (2, 18.0)), None); // window 1: mean 9, streak 1
        assert_eq!(s.check(20.0, (4, 19.0)), None); // window 2: mean 0.5, reset
        assert_eq!(s.check(30.0, (5, 22.0)), None); // mean 3, streak 1
        assert_eq!(s.check(35.0, (5, 22.0)), None); // inside window 4
        assert_eq!(s.check(42.0, (5, 22.0)), None); // no completions: no change
        assert_eq!(s.check(55.0, (7, 32.0)), Some(5.0)); // mean 5, streak 2
    }

    #[test]
    fn rejects_nonpositive_wait() {
        assert!(Saturation::new(0.0).is_err());
        assert!(Saturation::new(f64::INFINITY).is_err());
    }
}
