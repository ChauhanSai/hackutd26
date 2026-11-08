//! ONNX inference engine for Benoit using pure-Rust tract-onnx.

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

pub const PRIVATE_DIM: usize = 57;
pub const EVENT_DIM: usize = 38;
pub const EVENT_HISTORY_LEN: usize = 32;

pub type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct BenoitInferenceEngine {
    plan: RunnableModel,
}

impl BenoitInferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        log::info!("Loading Benoit ONNX model from: {}", model_path);

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("Failed to read ONNX file from: {}", model_path))?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, PRIVATE_DIM)))?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, EVENT_HISTORY_LEN, EVENT_DIM)),
            )?
            .into_optimized()
            .context("Failed to optimize ONNX graph")?
            .into_runnable()
            .context("Failed to build runnable execution plan")?;

        Ok(Self { plan: model })
    }

    /// Run forward inference:
    /// private_state: [57] floats
    /// event_history: [32, 38] floats
    /// Returns: (bid_fraction, play_threshold)
    pub fn predict(
        &self,
        private_state: &[f32; PRIVATE_DIM],
        event_history: &[f32; EVENT_HISTORY_LEN * EVENT_DIM],
    ) -> Result<(f32, f32)> {
        // Build private tensor: [1, 57]
        let priv_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, PRIVATE_DIM),
            private_state.to_vec(),
        )?
        .into();

        // Build events tensor: [1, 32, 38]
        let ev_tensor: Tensor = tract_ndarray::Array3::from_shape_vec(
            (1, EVENT_HISTORY_LEN, EVENT_DIM),
            event_history.to_vec(),
        )?
        .into();

        // Execute forward pass
        let outputs = self.plan.run(tvec!(priv_tensor.into(), ev_tensor.into()))?;

        // Extract [1, 3] or [1, 2] output tensor
        let output_view = outputs[0].to_array_view::<f32>()?;
        let bid_fraction = output_view[[0, 0]].clamp(0.0, 1.0);
        let play_threshold = output_view[[0, 1]].clamp(0.0, 1.0);

        Ok((bid_fraction, play_threshold))
    }
}

