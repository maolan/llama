//! Llama 3 large language model with Burn.
//!
//! This crate is a vendored copy of `llama-burn` from the
//! [tracel-ai/models](https://github.com/tracel-ai/models) repository, prepared
//! as a standalone publishable crate for use by `maolan-generate`.

pub(crate) mod cache;
pub mod llama;
pub mod pretrained;
pub mod sampling;
pub mod tokenizer;
mod transformer;

#[cfg(test)]
mod tests {
    pub type TestBackend = burn_ndarray::NdArray;
    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}
