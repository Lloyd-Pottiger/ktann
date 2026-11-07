//! Deterministic randomness shared by generated datasets and replay histories.

/// A replayable xorshift64 generator for model and fault histories; the
/// printed seed reproduces a failure.
pub struct Rng(pub u64);

impl Rng {
    /// Returns the next pseudo-random word.
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Returns the next word modulo `bound`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}
