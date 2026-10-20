//! Exact vacuum tensor projection through HEPKit's native Spenso reducer.
use crate::{Error, Result};
use feynkit_kinematics::Kinematics;
use feynkit_tensor::{TensorReducer, TensorReductionError};
use idenso::tensor::SymbolicTensor;
use std::collections::BTreeMap;
use symbolica::prelude::*;

type ProjectionKey = (Vec<usize>, Vec<usize>);
// Native coefficient numerator, denominator, and invariant tensor expression.
type Projection = Vec<(Atom, Atom, Atom)>;

// Equal rows represent indistinguishable vectors for this Gram polynomial.
// This only translates the public Gram interface to native vector identities;
// HEPKit owns all contraction orbits, pairings and projector inversion.
fn vector_classes(gram: &[Vec<Atom>]) -> (Vec<usize>, Vec<usize>) {
    let mut representatives = Vec::<usize>::new();
    let classes = (0..gram.len())
        .map(|i| {
            if let Some(class) = representatives.iter().position(|&j| gram[i] == gram[j]) {
                class
            } else {
                representatives.push(i);
                representatives.len() - 1
            }
        })
        .collect();
    (classes, representatives)
}

fn native_error(error: TensorReductionError) -> Error {
    match error {
        TensorReductionError::UnsupportedRank { .. }
        | TensorReductionError::PairingLimit { .. }
        | TensorReductionError::PairingProductLimit { .. }
        | TensorReductionError::OutputLimit { .. }
        | TensorReductionError::MultiplicityOverflow(_)
        | TensorReductionError::CounterOverflow => Error::Limit(error.to_string()),
        _ => Error::Unsupported(format!("native tensor projection: {error}")),
    }
}

/// Project a product `(hard_vector[i] . external_vector[i])` inside a
/// rotationally invariant vacuum integral. Gram rows identify repeated
/// vectors. Odd rank vanishes. Native HEPKit supports general rank 20 and
/// fully contracted single-hard-vector tensors through rank 32, with bounded
/// invariant output. Symbolic dimensions are substituted after projection.
#[derive(Clone, Debug)]
pub struct TensorProjector {
    dimension: Atom,
    projections: BTreeMap<ProjectionKey, Projection>,
}
impl TensorProjector {
    pub fn new(dimension: Atom) -> Self {
        Self {
            dimension,
            projections: BTreeMap::new(),
        }
    }
    pub fn project(
        &mut self,
        hard_gram: &[Vec<Atom>],
        external_gram: &[Vec<Atom>],
    ) -> Result<Atom> {
        let rank = hard_gram.len();
        if external_gram.len() != rank
            || hard_gram
                .iter()
                .chain(external_gram)
                .any(|r| r.len() != rank)
        {
            return Err(Error::InvalidInput("tensor Gram dimensions".into()));
        }
        if (0..rank).any(|i| {
            (0..i).any(|j| {
                hard_gram[i][j] != hard_gram[j][i] || external_gram[i][j] != external_gram[j][i]
            })
        }) {
            return Err(Error::InvalidInput(
                "tensor Gram matrices must be symmetric".into(),
            ));
        }
        if rank % 2 == 1 {
            return Ok(Atom::new());
        }
        if rank == 0 {
            return Ok(Atom::one());
        }
        if rank > 32 {
            return Err(Error::Limit(format!(
                "vacuum tensor rank {rank} exceeds 32"
            )));
        }
        let (hard_classes, hard_representatives) = vector_classes(hard_gram);
        let (external_classes, external_representatives) = vector_classes(external_gram);
        let key = (hard_classes, external_classes);
        let dimension = Atom::var(symbol!("symbolica_amflow::tensor_projection_dimension"));
        let kinematics = Kinematics::in_dimension(&dimension)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let hard_head = spenso::vector_symbol!("symbolica_amflow::tensor_projection_hard");
        let external_head = spenso::vector_symbol!("symbolica_amflow::tensor_projection_external");
        let dot = |left: &Atom, right: &Atom| {
            kinematics
                .scalar_product(left, right)
                .map_err(|error| Error::InvalidInput(error.to_string()))
        };
        if !self.projections.contains_key(&key) {
            let numerator =
                key.0
                    .iter()
                    .zip(&key.1)
                    .try_fold(Atom::one(), |a, (&hard, &external)| {
                        Ok::<_, Error>(
                            a * dot(&hard_head.call(hard), &external_head.call(external))?,
                        )
                    })?;
            let numerator =
                SymbolicTensor::infer(numerator).map_err(|error| native_error(error.into()))?;
            let reduced = TensorReducer::new(dimension.clone())
                .with_integrated_head(hard_head)
                .reduce(&numerator)
                .map_err(native_error)?;
            if !reduced.is_fully_contracted() {
                return Err(Error::Unsupported(
                    "native vacuum projector retained free indices".into(),
                ));
            }
            let projection = reduced
                .terms()
                .iter()
                .map(|term| {
                    let coefficient: RationalPolynomial<IntegerRing, u16> = term
                        .coefficient()
                        .try_to_rational_polynomial(&Q, &Z, None)
                        .map_err(|error| {
                            Error::Unsupported(format!("native projector coefficient: {error}"))
                        })?;
                    Ok((
                        coefficient.numerator.to_expression(),
                        coefficient.denominator.to_expression(),
                        term.tensor().expression().clone(),
                    ))
                })
                .collect::<Result<Projection>>()?;
            self.projections.insert(key.clone(), projection);
        }
        let mut gram = BTreeMap::new();
        for (head, representatives, values) in [
            (hard_head, &hard_representatives, hard_gram),
            (external_head, &external_representatives, external_gram),
        ] {
            for (i, &left) in representatives.iter().enumerate() {
                for (j, &right) in representatives.iter().enumerate() {
                    gram.insert(
                        dot(&head.call(i), &head.call(j))?,
                        values[left][right].clone(),
                    );
                }
            }
        }
        let dimension = BTreeMap::from([(dimension, self.dimension.clone())]);
        let mut value = Atom::new();
        for (numerator, denominator, tensor) in &self.projections[&key] {
            let denominator = crate::family::substitute(denominator, &dimension)
                .together()
                .cancel();
            if denominator.is_zero() {
                return Err(Error::Numerical(
                    "singular isotropic tensor dimension".into(),
                ));
            }
            value += crate::family::substitute(numerator, &dimension) / denominator
                * crate::family::substitute(tensor, &gram);
        }
        Ok(value.together().cancel())
    }
}
