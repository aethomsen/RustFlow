//! Profile generation and native interpolation for checkpointed campaign samples.
use serde::Deserialize;
use std::path::PathBuf;
use symbolica::prelude::*;
use symbolica_amflow::{epsilon::EpsilonFitProfile, *};

#[derive(Deserialize)]
struct Input {
    operation: String,
    digits: u32,
    leading: i32,
    last: i32,
    refinement: usize,
    #[serde(default)]
    sample_files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input_path = args.next().ok_or_else(|| {
        Error::InvalidInput("usage: campaign_epsilon INPUT.json OUTPUT.json".into())
    })?;
    let output_path = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::InvalidInput("missing output path".into()))?,
    );
    let input: Input = serde_json::from_reader(std::fs::File::open(input_path)?)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    let profile = EpsilonFitProfile::new(
        input.digits,
        40,
        input.leading,
        input.last,
        input.refinement,
    )?;
    let mut output = serde_json::json!({
        "digits": input.digits,
        "leading": input.leading,
        "last": input.last,
        "refinement": input.refinement,
        "working_digits": profile.working_digits,
        "radius": profile.radius.to_string(),
        "epsilon_samples": profile.samples.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "accuracy_verified": false,
    });
    match input.operation.as_str() {
        "profile" => {}
        "fit" => {
            if input.sample_files.len() != profile.samples.len() {
                return Err(Error::InvalidInput(
                    "sample count does not match profile".into(),
                ));
            }
            let p = Precision::decimal(profile.working_digits)?;
            let mut values = Vec::new();
            for (path, epsilon) in input.sample_files.iter().zip(&profile.samples) {
                let sample: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)
                    .map_err(|e| Error::InvalidInput(e.to_string()))?;
                if sample["epsilon"] != epsilon.to_string()
                    || sample["normalization"] != "FMFTandMATAD"
                    || sample["working_digits"].as_u64().unwrap_or(0)
                        < u64::from(profile.working_digits)
                    || sample["target_gamma_formula_used"] != false
                {
                    return Err(Error::InvalidInput(format!(
                        "sample provenance mismatch: {}",
                        path.display()
                    )));
                }
                let target = &sample["target"];
                values.push(
                    p.parse(
                        target[0].as_str().ok_or_else(|| {
                            Error::InvalidInput("missing target real string".into())
                        })?,
                        target[1].as_str().ok_or_else(|| {
                            Error::InvalidInput("missing target imaginary string".into())
                        })?,
                    )?,
                );
            }
            let fitted = fit_epsilon(&profile.samples, &values, input.leading, input.last, p)?;
            let coefficients = fitted
                .coefficients
                .iter()
                .map(|(power, value)| {
                    if !value.im.is_zero() {
                        return Err(Error::Numerical(
                            "unit-mass vacuum fit has an imaginary coefficient".into(),
                        ));
                    }
                    Ok(serde_json::json!([power, value.re.as_raw().to_string()]))
                })
                .collect::<Result<Vec<_>>>()?;
            output["coefficients"] = serde_json::Value::Array(coefficients);
            output["working_bits"] = serde_json::json!(fitted.working_bits);
            output["sample_files"] = serde_json::json!(input.sample_files);
        }
        _ => {
            return Err(Error::InvalidInput(
                "operation must be profile or fit".into(),
            ));
        }
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = output_path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&output).unwrap())?;
    std::fs::rename(tmp, output_path)?;
    Ok(())
}
