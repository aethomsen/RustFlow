//! Allocation admission for long numerical series; external process limits
//! remain necessary because these are conservative estimates, not allocator quotas.
use crate::{Error, Precision, Result};

pub(crate) const LEGACY_MAX_ORDER: usize = 10_000;
pub(crate) const MAX_ORDER: usize = 200_000;
pub(crate) const MAX_ESTIMATED_BYTES: u128 = 32 * 1024_u128.pow(3);

pub(crate) fn complex_bytes(p: Precision) -> u128 {
    2 * (u128::from(p.bits).div_ceil(8) + 64)
}

pub(crate) fn admit_bytes(order: usize, bytes: u128, operation: &str) -> Result<()> {
    if order > MAX_ORDER || bytes > MAX_ESTIMATED_BYTES {
        return Err(Error::Limit(format!(
            "{operation} exceeds {MAX_ORDER} series terms or the 32 GiB allocation estimate ({bytes} bytes)"
        )));
    }
    Ok(())
}

pub(crate) fn admit_complex(
    p: Precision,
    order: usize,
    values: u128,
    operation: &str,
) -> Result<()> {
    admit_bytes(order, values.saturating_mul(complex_bytes(p)), operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_rejects_overflow_and_oversized_arrays_without_allocating() {
        let p = Precision::decimal(24_000).unwrap();
        assert!(admit_complex(p, 27_000, 12 * 4 * 27_001, "Taylor chart").is_ok());
        assert!(matches!(
            admit_complex(p, 27_000, u128::MAX, "Taylor chart"),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            admit_complex(p, MAX_ORDER + 1, 1, "Taylor chart"),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            admit_complex(Precision { bits: u32::MAX }, 10_001, 10_002, "Taylor chart"),
            Err(Error::Limit(_))
        ));
    }
}
