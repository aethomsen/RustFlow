//! Native coherent W/Z and infinite-top Higgs-plus-jet observables.
//!
//! HEPKit owns effective-diagram generation and state sums; Idenso/Spenso own
//! tensor contraction. This module supplies the physical scalar products and
//! uses Symbolica to separate coupling orders and propagate input uncertainty.
use crate::{ComplexFloat, Error, Precision, Result, RunContext};
use feynkit_generator::{GenerationOptions, Process};
use feynkit_graph::FeynmanDiagram;
use feynkit_kinematics::Kinematics;
use feynkit_model::Model;
use idenso::color::ColorSimplifier;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use symbolica::prelude::*;

/// Eight normalized complex coefficients: W1..W4 followed by Z1..Z4.
/// Each allowance bounds both real and imaginary components conservatively.
/// The coefficients must be evaluated at the same W/Z masses and kinematics
/// as the supplied model parameters; this scalar container has no mass metadata.
pub struct HiggsJetFormFactors {
    pub values: [ComplexFloat; 8],
    pub absolute_errors: [Float; 8],
    pub provenance: String,
}

#[derive(Clone, Debug)]
pub struct HiggsJetObservable {
    pub value: Float,
    pub absolute_error: Float,
    pub arithmetic_change: Float,
    /// Relative digits conditional on the supplied form-factor allowances.
    pub verified_relative_digits: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct HiggsJetObservables {
    pub electroweak_squared: HiggsJetObservable,
    pub interference: HiggsJetObservable,
    pub effective_squared: HiggsJetObservable,
    pub working_bits: u32,
    pub input_provenance: String,
}

/// A kinematics-dependent native scalar kernel shared by repeated evaluations.
pub struct HiggsJetAmplitude {
    diagrams: Vec<Arc<FeynmanDiagram>>,
    expressions: [Atom; 3],
    coordinates: [Symbol; 3],
    form_factors: Vec<Symbol>,
    model_parameters: BTreeSet<Atom>,
}

fn native_error(e: impl std::fmt::Display) -> Error {
    Error::InvalidInput(format!("HEPKit Higgs-jet amplitude: {e}"))
}
fn ufo(name: &str) -> Atom {
    Atom::var(symbol!(&format!("UFO::{name}")))
}

/// Exact scalar HEFT coefficients in the effective model's tensor basis.
/// Scientific provenance: original plugin `ComputationFormFacHEFT/fortran_bridge_HEFT_ggHg.cpp`,
/// SHA256 e2df9f77089477730ca8b00ee591aa926bb6f242f1c7629880ab510357fad8f2.
/// These are kinematic rational formulas, not numerically fitted input data.
pub fn effective_form_factors(s: &Atom, t: &Atom, higgs_mass_squared: &Atom) -> [Atom; 4] {
    [
        Atom::num(-4) / t,
        Atom::num(4) / s,
        Atom::num(-4) / s,
        (Atom::num(-4) * (s.pow(2) + s * t + t.pow(2) - higgs_mass_squared * (s + t)))
            / (s * t * (s + t - higgs_mass_squared)),
    ]
}

impl HiggsJetAmplitude {
    /// Add the three effective gggH form-factor vertices to a Standard Model.
    ///
    /// Existing particles, parameters and interactions are retained. The added
    /// declarations contain symbolic tensors and couplings, not numerical loop
    /// results. HEPKit validates the resulting model through its native model
    /// interchange. Existing declaration names are never overwritten.
    pub fn with_form_factor_vertices(model: &Model) -> Result<Model> {
        let gluon = model.particle_by_pdg(21).map_err(native_error)?;
        let higgs = model.particle_by_pdg(25).map_err(native_error)?;
        if (gluon.spin, gluon.color, higgs.spin, higgs.color) != (3, 8, 1, 1) {
            return Err(native_error(
                "form-factor vertices require a gluon and a scalar Higgs",
            ));
        }
        for name in ["aS", "vev", "G", "MW", "gw", "sw", "cw"] {
            model.parameter(name).map_err(native_error)?;
        }
        let mut definition: serde_json::Value =
            serde_json::from_str(&model.to_json().map_err(native_error)?).map_err(native_error)?;
        let mut additions: serde_json::Value = serde_json::from_str(include_str!(
            "../../fixtures/gg-hg/form-factor-vertices.json"
        ))
        .map_err(native_error)?;
        for vertex in additions["vertex_rules"]
            .as_array_mut()
            .ok_or_else(|| native_error("invalid form-factor vertex declarations"))?
        {
            vertex["particles"] =
                serde_json::json!([gluon.name, gluon.name, gluon.name, higgs.name,]);
        }
        for section in [
            "orders",
            "parameters",
            "lorentz_structures",
            "couplings",
            "vertex_rules",
        ] {
            let existing = definition[section]
                .as_array_mut()
                .ok_or_else(|| native_error(format!("invalid model {section}")))?;
            let incoming = additions[section]
                .as_array()
                .ok_or_else(|| native_error(format!("invalid form-factor {section}")))?;
            for declaration in incoming {
                let name = declaration["name"]
                    .as_str()
                    .ok_or_else(|| native_error("unnamed form-factor declaration"))?;
                if existing.iter().any(|entry| entry["name"] == name) {
                    return Err(native_error(format!(
                        "cannot add form-factor {section}: {name} already exists"
                    )));
                }
                existing.push(declaration.clone());
            }
        }
        definition["name"] = format!("{} + Higgs-jet form factors", model.name()).into();
        Model::from_json(&serde_json::to_string(&definition).map_err(native_error)?)
            .map_err(native_error)
    }

    /// Reuse one already loaded effective UFO model, including its scoped Lorentz structures.
    /// Incoming two-gluon spin/color averages and all outgoing state sums are included.
    pub fn new(model: Arc<Model>, context: &RunContext) -> Result<Self> {
        context.emit(crate::Progress::Stage {
            name: "generating native Higgs-jet effective diagrams".into(),
        })?;
        let ff_names = ["GGGHEWWW", "GGGHEWZZ"]
            .into_iter()
            .flat_map(|tag| {
                (1..=4).flat_map(move |index| {
                    ["RE", "IM"]
                        .into_iter()
                        .map(move |component| format!("{tag}_ForFac{index}_{component}"))
                })
            })
            .collect::<Vec<_>>();
        for name in ff_names
            .iter()
            .cloned()
            .chain((1..=4).map(|i| format!("GGGH_HEFT_ForFac{i}")))
        {
            model.parameter(&name).map_err(native_error)?;
        }
        let process = Process::new([21_i64, 21], [21_i64, 25]).with_filters(
            vec![],
            Some(vec![
                "GGGHEWZZ".into(),
                "GGGHEWWW".into(),
                "GGGHHEFT".into(),
            ]),
            vec![],
        );
        let amplitude = process
            .generate_amplitude(
                model.clone(),
                &GenerationOptions::default()
                    .with_loop_count(0, 0)
                    .map_err(native_error)?
                    .threads(1),
                Default::default(),
            )
            .map_err(native_error)?;
        context.emit(crate::Progress::Stage {
            name: "summing native Higgs-jet spin and color states".into(),
        })?;
        let diagrams = amplitude.diagrams().to_vec();
        let summed = amplitude
            .squared()
            .map_err(native_error)?
            .sum_spins(
                &[0, 1, 2, 3],
                true,
                &Default::default(),
                &Default::default(),
            )
            .map_err(native_error)?
            .sum_colors(&[0, 1, 2, 3], true)
            .map_err(native_error)?;
        context.emit(crate::Progress::Stage {
            name: "contracting native Higgs-jet tensors".into(),
        })?;
        let algebra = idenso::tensor::AlgebraSettings {
            contract: idenso::tensor::AlgebraContraction::Dots,
            ..idenso::tensor::AlgebraSettings::hep()
        };
        // Reduce connected tensor work before distributing products. Expansion
        // exposes deferred closed scopes for the final native reduction.
        let scalar = summed
            .expression()
            .simplify_algebra(&algebra)
            .map_err(native_error)?
            .expanded(None, false)
            .map_err(native_error)?
            .simplify_algebra(&algebra)
            .map_err(native_error)?;
        if !scalar.contraction_complete() || !scalar.is_scalar() {
            return Err(Error::Unsupported(
                "Higgs-jet state sum retains unresolved tensor contractions".into(),
            ));
        }
        context.emit(crate::Progress::Stage {
            name: "preparing kinematics-dependent Higgs-jet scalar kernels".into(),
        })?;
        let coordinates = [
            symbol!("symbolica_amflow::higgs_jet_s"),
            symbol!("symbolica_amflow::higgs_jet_t"),
            symbol!("symbolica_amflow::higgs_jet_mass_squared"),
        ];
        let [s, t, h] = coordinates.map(Atom::var);
        let u = &h - &s - &t;
        let momenta = (0..4)
            .map(|i| feynkit_graph::symbols::external_momentum().call(i))
            .collect::<Vec<_>>();
        let products = [
            [Atom::Zero, &s / 2, -&t / 2, (&s + &t) / 2],
            [&s / 2, Atom::Zero, -&u / 2, (&s + &u) / 2],
            [-&t / 2, -&u / 2, Atom::Zero, (&s - &h) / 2],
            [(&s + &t) / 2, (&s + &u) / 2, (&s - &h) / 2, h.clone()],
        ];
        let mut kinematics = Kinematics::new();
        for (i, row) in products.iter().enumerate() {
            for (j, product) in row.iter().enumerate().skip(i) {
                kinematics = kinematics
                    .with_scalar_product(&momenta[i], &momenta[j], product.clone())
                    .map_err(native_error)?;
            }
        }
        let scalar = kinematics.apply(&scalar.into_expression().to_cof_dimension_invariants());
        let expression = model.expand_couplings(&scalar);
        let mut expression = model.expand_parameters(&expression).map_err(native_error)?;
        context.cancellation.check()?;
        let wildcard = symbol!("symbolica_amflow::higgs_jet_conjugate_");
        expression = expression
            .replace(function!(symbol!("spenso::conj"), wildcard))
            .with(Atom::var(wildcard).conj());
        let variables = ff_names.iter().map(|name| ufo(name)).collect::<Vec<_>>();
        for variable in variables
            .iter()
            .cloned()
            .chain((1..=4).map(|i| ufo(&format!("GGGH_HEFT_ForFac{i}"))))
        {
            expression = expression.replace(variable.conj()).with(variable);
        }
        let marker = Atom::var(symbol!("symbolica_amflow::higgs_jet_effective_marker"));
        let replacements = effective_form_factors(&s, &t, &h)
            .into_iter()
            .enumerate()
            .map(|(i, value)| (ufo(&format!("GGGH_HEFT_ForFac{}", i + 1)), &marker * value))
            .collect();
        let expression = crate::family::substitute(&expression, &replacements).expand();
        let mut expressions = [Atom::Zero, Atom::Zero, Atom::Zero];
        for (monomial, coefficient) in
            crate::coefficient::exact_coefficient_list(&expression, std::slice::from_ref(&marker))?
        {
            let degree = crate::coefficient::powers(&monomial, std::slice::from_ref(&marker))?[0];
            if !(0..=2).contains(&degree) || coefficient.contains(marker.as_view()) {
                return Err(Error::Unsupported(
                    "native Higgs-jet kernel is not quadratic in the effective coupling marker"
                        .into(),
                ));
            }
            expressions[degree as usize] += coefficient;
        }
        let reconstructed =
            &expressions[0] + &marker * &expressions[1] + marker.pow(2) * &expressions[2];
        if !(expression - reconstructed).expand().is_zero() {
            return Err(Error::Numerical(
                "exact Higgs-jet coupling-order reconstruction failed".into(),
            ));
        }
        let form_factors = variables
            .iter()
            .map(|a| match a.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        for (index, expression) in expressions.iter().enumerate() {
            for (monomial, coefficient) in
                crate::coefficient::exact_coefficient_list(expression, &variables)?
            {
                let exponents = crate::coefficient::powers(&monomial, &variables)?;
                if exponents.iter().any(|e| *e < 0)
                    || exponents.iter().map(|&e| i32::from(e)).sum::<i32>() != 2 - index as i32
                    || form_factors
                        .iter()
                        .any(|v| coefficient.get_all_symbols(false).contains(v))
                {
                    return Err(Error::Unsupported(
                        "native Higgs-jet kernel has unexpected form-factor degree".into(),
                    ));
                }
            }
        }
        let model_parameters = model.parameters().iter().map(|p| ufo(&p.name)).collect();
        Ok(Self {
            diagrams,
            expressions,
            coordinates,
            form_factors,
            model_parameters,
        })
    }
    pub fn diagrams(&self) -> &[Arc<FeynmanDiagram>] {
        &self.diagrams
    }
    /// EW square, interference, and HEFT square, in that order.
    pub fn expressions(&self) -> &[Atom; 3] {
        &self.expressions
    }
    pub fn coordinate_symbols(&self) -> [Symbol; 3] {
        self.coordinates
    }

    /// Evaluate at real physical kinematics with explicit exact model parameters.
    /// Returned errors are conditional on the supplied form-factor allowances;
    /// precision refinement estimates arithmetic error rather than proving it.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        s: &Atom,
        t: &Atom,
        higgs_mass_squared: &Atom,
        factors: &HiggsJetFormFactors,
        parameters: &BTreeMap<Atom, Atom>,
        digits: u32,
        guard_digits: u32,
        context: &RunContext,
    ) -> Result<HiggsJetObservables> {
        context.emit(crate::Progress::Stage {
            name: "evaluating coherent Higgs-jet observables and propagated errors".into(),
        })?;
        if digits == 0 || factors.provenance.trim().is_empty() {
            return Err(Error::InvalidInput(
                "positive accuracy and form-factor provenance required".into(),
            ));
        }
        let working = digits
            .checked_add(guard_digits.max(20))
            .ok_or_else(|| Error::Limit("amplitude precision overflow".into()))?;
        let low = Precision::decimal(working)?;
        let high = Precision::decimal(
            working
                .checked_add(32)
                .ok_or_else(|| Error::Limit("amplitude precision overflow".into()))?,
        )?;
        let mut substitutions = parameters.clone();
        for (key, value) in &substitutions {
            if !self.model_parameters.contains(key)
                || self.form_factors.iter().any(|&s| key == &Atom::var(s))
                || (1..=4).any(|i| key == &ufo(&format!("GGGH_HEFT_ForFac{i}")))
            {
                return Err(Error::InvalidInput(format!(
                    "unexpected or form-factor model parameter {key}"
                )));
            }
            crate::family::scalar_symbols(value.as_view(), &mut Default::default())?;
        }
        let u = higgs_mass_squared - s - t;
        let mut numerical = Vec::new();
        for atom in [s, t, higgs_mass_squared, &u] {
            crate::family::scalar_symbols(atom.as_view(), &mut Default::default())?;
            let v = high.eval(atom, &Default::default())?;
            if !high.finite(&v) || !v.im.is_zero() {
                return Err(Error::InvalidInput(
                    "s, t and Higgs mass squared must be exact real constants".into(),
                ));
            }
            numerical.push(v.re);
        }
        if numerical[2] <= high.real(0)
            || numerical[0] <= numerical[2]
            || numerical[1] >= high.real(0)
            || numerical[3] >= high.real(0)
        {
            return Err(Error::InvalidInput(
                "Higgs-jet observables require s > MH² > 0, t < 0, u < 0".into(),
            ));
        }
        if let Some(mass) = substitutions.get(&ufo("MH")) {
            if !(mass.pow(2) - higgs_mass_squared)
                .together()
                .cancel()
                .is_zero()
            {
                return Err(Error::InvalidInput(
                    "model MH does not match the kinematic Higgs mass".into(),
                ));
            }
        } else {
            substitutions.insert(ufo("MH"), higgs_mass_squared.pow(Rational::from((1, 2))));
        }
        if substitutions
            .get(&ufo("ZERO"))
            .is_some_and(|a| !a.is_zero())
        {
            return Err(Error::InvalidInput("model ZERO must be zero".into()));
        }
        substitutions.insert(ufo("ZERO"), Atom::Zero);
        substitutions.extend(
            self.coordinates
                .into_iter()
                .zip([s, t, higgs_mass_squared])
                .map(|(symbol, value)| (Atom::var(symbol), value.clone())),
        );
        let mut values = ahash::HashMap::default();
        let mut errors = Vec::new();
        for (index, factor) in factors.values.iter().enumerate() {
            let error = &factors.absolute_errors[index];
            if !high.finite(factor) || !error.is_finite() || *error < high.real(0) {
                return Err(Error::InvalidInput(
                    "finite form factors and nonnegative finite allowances required".into(),
                ));
            }
            for (component, value) in [&factor.re, &factor.im].into_iter().enumerate() {
                values.insert(
                    Atom::var(self.form_factors[index * 2 + component]),
                    ComplexFloat::new(value.clone(), high.real(0)),
                );
                errors.push(ComplexFloat::new(error.clone(), high.real(0)));
            }
        }
        let mut results = Vec::new();
        for expression in &self.expressions {
            context.cancellation.check()?;
            let expression = crate::family::substitute(expression, &substitutions);
            let coarse = low.eval(&expression, &values)?;
            let value = high.eval(&expression, &values)?;
            if !high.finite(&value) || !low.finite(&coarse) {
                return Err(Error::Numerical("nonfinite Higgs-jet observable".into()));
            }
            let change = high.norm(&high.sub(&value, &coarse));
            let magnitude = high.norm(&value);
            let scale = if magnitude > high.real(1) {
                magnitude
            } else {
                high.real(1)
            };
            let reserve = high
                .mul(
                    &ComplexFloat::new(scale, high.real(0)),
                    &ComplexFloat::new(high.tolerance(working.saturating_sub(10)), high.real(0)),
                )
                .re;
            if change > reserve
                || high.norm(&ComplexFloat::new(value.im.clone(), high.real(0))) > reserve
            {
                return Err(Error::InsufficientPrecision {
                    minimum_bits: high.bits,
                    context: "Higgs-jet scalar reality or independent precision check failed"
                        .into(),
                });
            }
            let mut allowance = ComplexFloat::new(reserve, high.real(0));
            for (i, &variable) in self.form_factors.iter().enumerate() {
                let derivative = expression.derivative(variable);
                let gradient = high.norm(&high.eval(&derivative, &values)?);
                allowance = high.add(
                    &allowance,
                    &high.mul(&ComplexFloat::new(gradient, high.real(0)), &errors[i]),
                );
                for (j, &other) in self.form_factors.iter().enumerate() {
                    let hessian = high.norm(&high.eval(&derivative.derivative(other), &values)?);
                    allowance = high.add(
                        &allowance,
                        &high.scale(
                            &high.mul(
                                &ComplexFloat::new(hessian, high.real(0)),
                                &high.mul(&errors[i], &errors[j]),
                            ),
                            1,
                            2,
                        ),
                    );
                }
            }
            let magnitude = high.norm(&value);
            let verified_relative_digits = if magnitude == high.real(0) {
                None
            } else {
                let relative = high
                    .div(&allowance, &ComplexFloat::new(magnitude, high.real(0)))
                    .re;
                Some(
                    (1..=working)
                        .take_while(|&d| relative <= high.tolerance(d))
                        .last()
                        .unwrap_or(0),
                )
            };
            results.push(HiggsJetObservable {
                value: value.re,
                absolute_error: allowance.re,
                arithmetic_change: change,
                verified_relative_digits,
            });
        }
        let mut results = results.into_iter();
        Ok(HiggsJetObservables {
            electroweak_squared: results.next().unwrap(),
            interference: results.next().unwrap(),
            effective_squared: results.next().unwrap(),
            working_bits: high.bits,
            input_provenance: factors.provenance.clone(),
        })
    }
}
