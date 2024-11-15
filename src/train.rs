use crate::data::Dataset;
use crate::model::{MinGRUConfig, MinGRULM};
use anyhow;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};

pub struct TrainConfig {
    num_batches: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    validate_every: usize,
    prime_length: usize,
    generate_every: usize,
    generate_length: usize,
    seq_len: usize,
    temperature: f32,
    threshold: Option<f32>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_batches: 1e5 as usize,
            batch_size: 4,
            grad_accum: 4,
            learning_rate: 1e-4,
            validate_every: 100,
            prime_length: 128,
            generate_every: 500,
            generate_length: 512,
            seq_len: 512,
            temperature: 1.0,
            threshold: Some(0.9),
        }
    }
}

impl TrainConfig {
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

fn decode_token(token: u8) -> char {
    let tok = token.max(32);
    char::from_u32(tok as u32).unwrap_or(' ')
}
fn decode_tokens(tokens: Vec<u8>) -> String {
    tokens
        .into_iter()
        .map(|t| decode_token(t))
        .collect::<String>()
}

fn gumbel_noise(t: &Tensor) -> Result<Tensor> {
    let noise = t.rand_like(0.0, 1.0)?;
    let noise = noise.log()?.affine(-1., 0.)?;
    noise.log()?.affine(-1., 0.)
}

fn gumbel_sample(t: &Tensor, temperature: f32, dim: D) -> Result<Tensor> {
    let temp = temperature.max(1e-10) as f64;
    let noise = gumbel_noise(t)?;
    t.affine(1. / temp, 0.)?
        .add(&noise)?
        .argmax_keepdim(dim)?
        .to_dtype(DType::U8)
}

fn top_k(logits: &Tensor, threshold: Option<f32>) -> Result<Tensor> {
    let last_dim = logits.dim(D::Minus1)?;
    let k = ((1.0 - threshold.unwrap_or(0.9)) * last_dim as f32).ceil() as usize;
    let (sorted_logits, sorted_idx) = logits.sort_last_dim(false)?;
    let probs = logits.ones_like()?.affine(f64::MIN, 0.)?;
    probs.scatter_add(
        &sorted_idx.narrow(D::Minus1, 0, k)?,
        &sorted_logits.narrow(D::Minus1, 0, k)?,
        D::Minus1,
    )
}

fn base_decoding(
    model: &MinGRULM,
    prompt: &Tensor,
    seq_len: usize,
    temperature: f32,
    threshold: Option<f32>,
) -> Result<Tensor> {
    let prompt_seq_len = prompt.dim(D::Minus1)?;
    let mut out = prompt.clone();
    let sample_n_times = (seq_len - prompt_seq_len).max(0);
    let mut prev_hiddens: Option<Vec<Tensor>> = None;
    for i in 0..sample_n_times {
        let res = model.forward(&out, false, prev_hiddens)?;
        let dim1_size = res.0.dim(1)?;
        let logits = res.0.narrow(1, dim1_size - 1, 1)?;
        if model.can_cache() {
            prev_hiddens = Some(res.1);
        } else {
            prev_hiddens = None;
        }
        let logits = top_k(&logits, threshold)?;
        let sample = gumbel_sample(&logits, temperature, D::Minus1)?;
        out = Tensor::cat(&[&out, &sample.squeeze(0)?], D::Minus1)?;
    }
    out.i((.., prompt_seq_len..))
}

pub fn training_loop(m: &mut Dataset, cfg: &TrainConfig) -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model_cfg = MinGRUConfig::default();
    let model = MinGRULM::new(model_cfg, vs.clone())?;

    let mut optim = AdamW::new_lr(varmap.all_vars(), cfg.learning_rate)?;

    let mut acc_loss = Tensor::zeros((), DType::F32, &dev)?;
    for i in 0..cfg.num_batches {
        let batch_input = m.get_batch("train", cfg.batch_size)?.to_device(&dev)?;
        let (loss, _) = model.forward(&batch_input, true, None)?;
        acc_loss = acc_loss.add(&loss)?;

        if i % cfg.grad_accum == 0 {
            let train_loss = Tensor::new(1. / (i * cfg.grad_accum) as f32, &dev)?;
            println!("Training loss: {}", train_loss.mul(&acc_loss)?);
            optim.backward_step(&acc_loss)?;
            // reset loss for accumulate grad batch
            acc_loss = Tensor::zeros((), DType::F32, &dev)?;
        }

        // one validation step
        if i % cfg.validate_every == 0 {
            let val_batch = m.get_batch("val", cfg.batch_size)?.to_device(&dev)?;
            let (val_loss, _) = model.forward(&val_batch, true, None)?;
            println!("Validation loss: {}", val_loss.to_vec0::<f32>()?);
        }

        // generate some random sentence from validation
        if i % cfg.generate_every == 0 {
            let inp = m.get_batch("val", 1)?;
            let inp = inp.narrow(D::Minus1, 0, cfg.prime_length)?;
            let prime = decode_tokens(inp.squeeze(0)?.to_vec1()?);
            println!("INPUT: {}", prime);

            let sampled = base_decoding(
                &model,
                &inp,
                cfg.generate_length,
                cfg.temperature,
                cfg.threshold,
            )?;
            println!("sampled: {:?}", sampled.dims());
            let decoded_output = decode_tokens(sampled.squeeze(0)?.to_vec1()?);
            println!("OUTPUT: {}", decoded_output);
        }
    }

    Ok(())
}
