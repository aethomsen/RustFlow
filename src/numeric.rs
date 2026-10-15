#![allow(clippy::needless_range_loop)] // Matrix row/column indexing mirrors the recurrence equations.
//! Fixed working-precision operations on Symbolica's selected arbitrary-precision numbers.
//!
//! Symbolica's ordinary Float arithmetic tracks significant bits. ODE recurrences
//! instead round each operation to an explicit working precision; accuracy is
//! assessed separately by recomputation, never inferred from that precision.
use crate::{Error, Result};
use std::{cell::RefCell, collections::VecDeque};
use symbolica::domains::float::{Complex, Float, RoundingDirection::Nearest};
use symbolica::prelude::*;

pub type ComplexFloat = Complex<Float>;

thread_local! {
    // Recursive boundaries repeatedly request the same gamma arguments at a
    // fixed precision. Keep a small per-worker cache; separate precisions never
    // share results during independent numerical refinement.
    static GAMMA_VALUES: RefCell<VecDeque<(u32, Float, ComplexFloat)>> = const {
        RefCell::new(VecDeque::new())
    };
}

const GAMMA_CACHE_MAX_ENTRIES: usize = 64;
const GAMMA_CACHE_MAX_BYTES: u64 = 8 * 1024 * 1024;

fn gamma_cache_entry_bytes(bits: u32) -> u64 {
    // One argument and the real/imaginary components of the value, plus
    // conservative float/container bookkeeping. The key stores working bits.
    3 * (u64::from(bits).div_ceil(8) + 64)
}

fn remember_gamma(bits: u32, argument: Float, value: ComplexFloat) {
    let bytes = gamma_cache_entry_bytes(bits);
    if bytes > GAMMA_CACHE_MAX_BYTES {
        return;
    }
    GAMMA_VALUES.with(|cache| {
        let mut cache = cache.borrow_mut();
        let mut retained = cache
            .iter()
            .map(|(bits, _, _)| gamma_cache_entry_bytes(*bits))
            .sum::<u64>();
        while cache.len() >= GAMMA_CACHE_MAX_ENTRIES || retained + bytes > GAMMA_CACHE_MAX_BYTES {
            let (bits, _, _) = cache.pop_front().unwrap();
            retained -= gamma_cache_entry_bytes(bits);
        }
        cache.push_back((bits, argument, value));
    });
}

#[cfg(test)]
thread_local! {
    static GAMMA_REQUEST_COUNTS: RefCell<Option<std::collections::HashMap<(u32, Float), usize>>> = const { RefCell::new(None) };
    static GAMMA_EVALUATIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[derive(Clone, Copy, Debug)]
pub struct Precision {
    pub bits: u32,
}

impl Precision {
    /// Sample the physical midpoint without discarding higher-precision input
    /// coordinates before averaging. This need not be an exact dyadic midpoint
    /// when the endpoints have widely separated binary exponents.
    pub(crate) fn coordinate_midpoint(
        &self,
        a: &ComplexFloat,
        b: &ComplexFloat,
    ) -> Result<ComplexFloat> {
        let bits = [
            self.bits,
            a.re.prec(),
            a.im.prec(),
            b.re.prec(),
            b.im.prec(),
        ]
        .into_iter()
        .max()
        .unwrap()
        .checked_add(1)
        .ok_or_else(|| Error::Limit("midpoint coordinate precision overflow".into()))?;
        let p = Self { bits };
        Ok(p.scale(&p.add(a, b), 1, 2))
    }

    /// Raise a decimal working precision using an integer bit-count hint.
    /// The conversion is conservative and shares `decimal`'s 3322/1000 scale.
    pub(crate) fn refinement_digits(current: u32, minimum_bits: u32) -> Result<u32> {
        let suggested = u64::from(minimum_bits)
            .checked_mul(1000)
            .and_then(|v| v.checked_add(3321))
            .map(|v| v / 3322 + 1)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Error::Limit("precision retry hint overflow".into()))?;
        current
            .checked_add(20)
            .map(|next| next.max(suggested))
            .ok_or_else(|| Error::Limit("precision refinement overflow".into()))
    }

    pub fn decimal(digits: u32) -> Result<Self> {
        let bits = digits
            .checked_mul(3322)
            .and_then(|v| v.checked_add(999))
            .map(|v| v / 1000 + 16)
            .ok_or_else(|| Error::InvalidInput("precision overflow".into()))?;
        if digits == 0 {
            return Err(Error::InvalidInput("precision must be positive".into()));
        }
        Ok(Self { bits })
    }
    pub fn real(&self, v: i64) -> Float {
        Float::with_val(self.bits, v)
    }
    pub fn zero(&self) -> ComplexFloat {
        self.i(0)
    }
    pub fn i(&self, v: i64) -> ComplexFloat {
        Complex::new(self.real(v), self.real(0))
    }
    pub fn complex(&self, re: i64, im: i64) -> ComplexFloat {
        Complex::new(self.real(re), self.real(im))
    }
    pub fn parse(&self, re: &str, im: &str) -> Result<ComplexFloat> {
        Ok(Complex::new(
            Float::parse(re, Some(self.bits)).map_err(Error::InvalidInput)?,
            Float::parse(im, Some(self.bits)).map_err(Error::InvalidInput)?,
        ))
    }
    pub fn rational(&self, value: &Rational) -> ComplexFloat {
        Complex::new(value.to_multi_prec_float(self.bits), self.real(0))
    }
    pub fn round(&self, a: &ComplexFloat) -> ComplexFloat {
        Complex::new(
            Float::with_val(self.bits, a.re.as_raw()),
            Float::with_val(self.bits, a.im.as_raw()),
        )
    }
    pub fn add(&self, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
        Complex::new(
            a.re.add_round(&b.re, self.bits, Nearest),
            a.im.add_round(&b.im, self.bits, Nearest),
        )
    }
    pub fn neg(&self, a: &ComplexFloat) -> ComplexFloat {
        Complex::new(-a.re.clone(), -a.im.clone())
    }
    pub fn sub(&self, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
        Complex::new(
            a.re.sub_round(&b.re, self.bits, Nearest),
            a.im.sub_round(&b.im, self.bits, Nearest),
        )
    }
    pub fn mul(&self, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
        // Keep the same rounded nonzero products as the general formula.
        // Real coefficients occur throughout the Taylor and Frobenius solves;
        // multiplying their exact zero imaginary part only allocates temporaries.
        // Nonfinite operands retain the general IEEE zero-times-infinity behavior.
        if b.im.is_zero() && self.finite(a) && self.finite(b) {
            return Complex::new(
                a.re.mul_round(&b.re, self.bits, Nearest),
                a.im.mul_round(&b.re, self.bits, Nearest),
            );
        }
        if a.im.is_zero() && self.finite(a) && self.finite(b) {
            return Complex::new(
                a.re.mul_round(&b.re, self.bits, Nearest),
                a.re.mul_round(&b.im, self.bits, Nearest),
            );
        }
        let ac = a.re.mul_round(&b.re, self.bits, Nearest);
        let bd = a.im.mul_round(&b.im, self.bits, Nearest);
        let ad = a.re.mul_round(&b.im, self.bits, Nearest);
        let bc = a.im.mul_round(&b.re, self.bits, Nearest);
        Complex::new(
            ac.sub_round(&bd, self.bits, Nearest),
            ad.add_round(&bc, self.bits, Nearest),
        )
    }
    pub fn div(&self, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
        if b.im.is_zero() && !b.re.is_zero() && self.finite(a) && self.finite(b) {
            // Divide each component directly by the real denominator. This
            // gives the correctly rounded quotient without intermediate
            // products or the extra rounding of the conjugate formula.
            return Complex::new(
                a.re.div_round(&b.re, self.bits, Nearest),
                a.im.div_round(&b.re, self.bits, Nearest),
            );
        }
        let conj = Complex::new(b.re.clone(), -b.im.clone());
        let top = self.mul(a, &conj);
        let den = self.mul(b, &conj).re;
        Complex::new(
            top.re.div_round(&den, self.bits, Nearest),
            top.im.div_round(&den, self.bits, Nearest),
        )
    }
    pub fn scale(&self, a: &ComplexFloat, n: i64, d: i64) -> ComplexFloat {
        if n == 1 && d != 0 && self.finite(a) {
            return self.div(a, &self.i(d));
        }
        self.div(&self.mul(a, &self.i(n)), &self.i(d))
    }
    pub fn norm(&self, a: &ComplexFloat) -> Float {
        if a.im.is_zero() && self.finite(a) {
            return Float::with_val(self.bits, a.re.as_raw()).norm();
        }
        if self.finite(a) {
            let real_square = a.re.mul_round(&a.re, self.bits, Nearest);
            let imaginary_square = a.im.mul_round(&a.im, self.bits, Nearest);
            return real_square
                .add_round(&imaginary_square, self.bits, Nearest)
                .sqrt();
        }
        self.mul(a, &Complex::new(a.re.clone(), -a.im.clone()))
            .re
            .sqrt()
    }
    pub fn exp(&self, a: &ComplexFloat) -> ComplexFloat {
        self.round(&self.round(a).exp())
    }
    pub fn log(&self, a: &ComplexFloat) -> ComplexFloat {
        self.round(&self.round(a).log())
    }
    pub fn pow(&self, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
        self.exp(&self.mul(b, &self.log(a)))
    }
    pub fn powi(&self, a: &ComplexFloat, n: i64) -> ComplexFloat {
        let mut power = n.unsigned_abs();
        let mut base = self.round(a);
        let mut out = self.i(1);
        while power != 0 {
            if power & 1 != 0 {
                out = self.mul(&out, &base);
            }
            power >>= 1;
            if power != 0 {
                base = self.mul(&base, &base);
            }
        }
        if n < 0 {
            self.div(&self.i(1), &out)
        } else {
            out
        }
    }
    /// Evaluate Gamma through Symbolica's selected arbitrary-precision backend.
    /// Keep the argument symbolic until evaluation so normalization does not
    /// select a different precision for the numeric function.
    pub fn gamma_real(&self, a: &Float) -> Result<ComplexFloat> {
        if !a.is_finite() {
            return Err(Error::Numerical("nonfinite gamma argument".into()));
        }
        let rounded = Float::with_val(self.bits, a.as_raw());
        #[cfg(test)]
        GAMMA_REQUEST_COUNTS.with(|counts| {
            if let Some(counts) = counts.borrow_mut().as_mut() {
                *counts.entry((self.bits, rounded.clone())).or_default() += 1;
            }
        });
        if let Some(value) = GAMMA_VALUES.with(|cache| {
            let mut cache = cache.borrow_mut();
            let index = cache
                .iter()
                .position(|(bits, argument, _)| *bits == self.bits && argument == &rounded)?;
            let entry = cache.remove(index).unwrap();
            let value = entry.2.clone();
            cache.push_back(entry);
            Some(value)
        }) {
            return Ok(value);
        }
        #[cfg(test)]
        GAMMA_EVALUATIONS.with(|evaluations| {
            if let Some(count) = evaluations.get() {
                evaluations.set(Some(count + 1));
            }
        });
        let argument = Atom::var(symbol!("symbolica_amflow::gamma_argument"));
        let expression = symbolica::transcendental::gamma().call(&argument);
        let parameters = ahash::HashMap::from_iter([(argument, rounded.clone())]);
        let value: Float = expression
            .evaluate_with_prec(&parameters, self.bits)
            .map_err(|error| Error::Numerical(format!("gamma evaluation: {error}")))?;
        if !value.is_finite() {
            return Err(Error::Numerical(
                "gamma evaluation reached a pole or exceeded the numeric backend's range".into(),
            ));
        }
        let result = Complex::new(Float::with_val(self.bits, value.as_raw()), self.real(0));
        remember_gamma(self.bits, rounded, result.clone());
        Ok(result)
    }
    pub fn tolerance(&self, digits: u32) -> Float {
        self.powi(&self.i(10), -(digits as i64)).re
    }
    pub fn finite(&self, a: &ComplexFloat) -> bool {
        a.re.is_finite() && a.im.is_finite()
    }
    pub fn close(&self, a: &ComplexFloat, b: &ComplexFloat, digits: u32) -> bool {
        if !self.finite(a) || !self.finite(b) {
            return false;
        }
        let norm = self.norm(a);
        let scale = if norm > self.real(1) {
            norm
        } else {
            self.real(1)
        };
        self.norm(&self.sub(a, b)) <= self.tolerance(digits) * scale
    }
    pub fn eval(
        &self,
        a: &Atom,
        values: &ahash::HashMap<Atom, ComplexFloat>,
    ) -> Result<ComplexFloat> {
        let mut values = values.clone();
        values.insert(
            Atom::var(crate::family::imaginary_parameter()),
            self.complex(0, 1),
        );
        let value = a
            .evaluate_with_prec(&values, self.bits)
            .map_err(|e| Error::Numerical(e.to_string()))?;
        if !self.finite(&value) {
            return Err(Error::Numerical(format!("nonfinite evaluation of {a}")));
        }
        Ok(self.round(&value))
    }
}

#[cfg(test)]
#[path = "numeric/precision_tests.rs"]
mod precision_tests;

/// Pivoted elimination, with all arithmetic at the chosen working precision.
pub fn solve(
    p: Precision,
    mut a: Vec<Vec<ComplexFloat>>,
    mut b: Vec<ComplexFloat>,
) -> Result<Vec<ComplexFloat>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|r| r.len() != n) {
        return Err(Error::InvalidInput("linear system dimensions".into()));
    }
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| {
                p.norm(&a[i][k])
                    .partial_cmp(&p.norm(&a[j][k]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        if p.norm(&a[pivot][k]) == p.real(0) {
            return Err(Error::Numerical("singular matching matrix".into()));
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = p.div(&a[i][k], &a[k][k]);
            for j in k..n {
                a[i][j] = p.sub(&a[i][j], &p.mul(&factor, &a[k][j]));
            }
            b[i] = p.sub(&b[i], &p.mul(&factor, &b[k]));
        }
    }
    let mut x = vec![p.zero(); n];
    for i in (0..n).rev() {
        let mut v = b[i].clone();
        for (j, xj) in x.iter().enumerate().skip(i + 1) {
            v = p.sub(&v, &p.mul(&a[i][j], xj));
        }
        x[i] = p.div(&v, &a[i][i]);
        if !p.finite(&x[i]) {
            return Err(Error::Numerical("nonfinite linear solution".into()));
        }
    }
    Ok(x)
}

pub(crate) fn solve_constraints(
    p: Precision,
    mut a: Vec<Vec<ComplexFloat>>,
    mut b: Vec<ComplexFloat>,
    columns: usize,
) -> Result<Vec<ComplexFloat>> {
    if a.len() != b.len() || a.iter().any(|r| r.len() != columns) {
        return Err(Error::InvalidInput("constraint dimensions".into()));
    }
    let mut row = 0;
    let mut pivots = Vec::new();
    let tolerance = p.tolerance(p.bits / 5);
    for col in 0..columns {
        let Some(pivot) = (row..a.len())
            .max_by(|&i, &j| p.norm(&a[i][col]).partial_cmp(&p.norm(&a[j][col])).unwrap())
        else {
            break;
        };
        if p.norm(&a[pivot][col]) <= tolerance {
            continue;
        }
        a.swap(row, pivot);
        b.swap(row, pivot);
        let divisor = a[row][col].clone();
        for j in col..columns {
            a[row][j] = p.div(&a[row][j], &divisor);
        }
        b[row] = p.div(&b[row], &divisor);
        for i in 0..a.len() {
            if i != row {
                let factor = a[i][col].clone();
                for j in col..columns {
                    a[i][j] = p.sub(&a[i][j], &p.mul(&factor, &a[row][j]));
                }
                b[i] = p.sub(&b[i], &p.mul(&factor, &b[row]));
            }
        }
        pivots.push(col);
        row += 1;
    }
    if pivots.len() != columns {
        return Err(Error::IncompleteReduction(format!(
            "boundary data fix {} of {columns} constants",
            pivots.len()
        )));
    }
    if b.iter().skip(row).any(|v| p.norm(v) > tolerance) {
        return Err(Error::Numerical(
            "inconsistent asymptotic boundary constraints".into(),
        ));
    }
    let mut out = vec![p.zero(); columns];
    for (i, &j) in pivots.iter().enumerate() {
        out[j] = b[i].clone();
    }
    Ok(out)
}

/// Reusable block-triangular LU factorization for Frobenius recurrences.
type BlockFactors = (Vec<usize>, Vec<Vec<ComplexFloat>>, Vec<usize>);

pub(crate) struct BlockSolve {
    p: Precision,
    matrix: Vec<Vec<ComplexFloat>>,
    factors: Vec<BlockFactors>,
}
impl BlockSolve {
    pub(crate) fn new(
        p: Precision,
        matrix: Vec<Vec<ComplexFloat>>,
        blocks: &[Vec<usize>],
    ) -> Result<Self> {
        let n = matrix.len();
        let mut owners = vec![usize::MAX; n];
        for (k, block) in blocks.iter().enumerate() {
            for &i in block {
                owners[i] = k;
            }
        }
        for (i, row) in matrix.iter().enumerate() {
            for (j, a) in row.iter().enumerate() {
                if owners[j] > owners[i] && *a != p.zero() {
                    return Err(Error::Numerical("invalid triangular block order".into()));
                }
            }
        }
        let mut factors = Vec::new();
        for block in blocks {
            let mut lu = block
                .iter()
                .map(|&i| {
                    block
                        .iter()
                        .map(|&j| matrix[i][j].clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let size = block.len();
            let mut permutation = (0..size).collect::<Vec<_>>();
            for k in 0..size {
                let pivot = (k..size)
                    .max_by(|&i, &j| p.norm(&lu[i][k]).partial_cmp(&p.norm(&lu[j][k])).unwrap())
                    .unwrap();
                if lu[pivot][k] == p.zero() {
                    return Err(Error::Numerical("singular Frobenius block".into()));
                }
                lu.swap(k, pivot);
                permutation.swap(k, pivot);
                for i in k + 1..size {
                    if lu[i][k] == p.zero() {
                        continue;
                    }
                    lu[i][k] = p.div(&lu[i][k], &lu[k][k]);
                    for j in k + 1..size {
                        lu[i][j] = p.sub(&lu[i][j], &p.mul(&lu[i][k], &lu[k][j]));
                    }
                }
            }
            factors.push((block.clone(), lu, permutation));
        }
        Ok(Self { p, matrix, factors })
    }
    pub(crate) fn solve(&self, rhs: &[ComplexFloat]) -> Vec<ComplexFloat> {
        let p = self.p;
        let mut out = vec![p.zero(); rhs.len()];
        for (block, lu, permutation) in &self.factors {
            let b = block
                .iter()
                .map(|&i| {
                    self.matrix[i]
                        .iter()
                        .zip(&out)
                        .fold(rhs[i].clone(), |a, (m, x)| {
                            if *m == p.zero() || *x == p.zero() {
                                a
                            } else {
                                p.sub(&a, &p.mul(m, x))
                            }
                        })
                })
                .collect::<Vec<_>>();
            let mut x = permutation
                .iter()
                .map(|&i| b[i].clone())
                .collect::<Vec<_>>();
            for i in 0..x.len() {
                for j in 0..i {
                    x[i] = p.sub(&x[i], &p.mul(&lu[i][j], &x[j]));
                }
            }
            for i in (0..x.len()).rev() {
                for j in i + 1..x.len() {
                    x[i] = p.sub(&x[i], &p.mul(&lu[i][j], &x[j]));
                }
                x[i] = p.div(&x[i], &lu[i][i]);
            }
            for (&i, x) in block.iter().zip(x) {
                out[i] = x;
            }
        }
        out
    }
}
