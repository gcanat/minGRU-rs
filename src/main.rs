use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{ops, Linear, Module, VarBuilder};

pub fn heinsen_associative_scan_log(log_coeffs: &Tensor, log_values: &Tensor) -> Result<Tensor> {
    let a_star = log_coeffs.cumsum(1)?;
    let log_h0_plus_b_star = log_values.sub(&a_star)?.exp()?.cumsum(1)?.log()?;
    let log_h = a_star.add(&log_h0_plus_b_star)?;
    Ok(log_h.exp()?)
}

pub fn g(x: &Tensor) -> Result<Tensor> {
    x.lt(0_u32)?
        .where_cond(&ops::sigmoid(&x)?, &x.add(&Tensor::new(0.5, x.device())?)?)
}

pub fn softplus(x: &Tensor, beta: f32, threshold: f32) -> Result<Tensor> {
    let beta = Tensor::new(beta, x.device())?;
    let mask = x.mul(&beta)?.gt(threshold)?;
    let res = x.mul(&beta)?.exp()?.div(&beta)?;
    mask.where_cond(&x, &res)
}

pub fn log_g(x: &Tensor) -> Result<Tensor> {
    x.lt(0_u32)?.where_cond(
        &softplus(&x, 1.0, 20.0)?.mul(&Tensor::new(-1.0, x.device())?)?,
        &x.relu()?.add(&Tensor::new(0.5, x.device())?)?.log()?,
    )
}

pub fn lerp(start: &Tensor, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
    start.add(&end.sub(start)?.mul(weight)?)
}

/// MinGRU log-space as in appendix B.3.1
struct MinGRU {
    to_hidden_and_gate: Linear,
    to_out: Linear,
    exp_factor: f32,
}

impl MinGRU {
    fn new(dim: usize, exp_factor: f32) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let dim_inner = (dim as f32 * exp_factor).round() as usize;

        // first linear layer
        let weight = Tensor::randn(0_f32, 1.0, (dim, dim_inner * 2), &device)?;
        let to_hidden_and_gate = Linear::new(weight, None);

        // second linear layer
        let weight = Tensor::randn(0_f32, 1.0, (dim_inner, dim), &device)?;
        let to_out = Linear::new(weight, None);

        Ok(MinGRU {
            to_hidden_and_gate,
            to_out,
            exp_factor,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        prev_hidden: Option<&Tensor>,
        return_next_prev_hidden: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let dims = x.dims();
        let seq_len = dims[1];
        let hid_n_gate = self.to_hidden_and_gate.forward(x)?.chunk(2, D::Minus1)?;

        let mut out: Tensor;

        match seq_len {
            1 => {
                let hidden = g(&hid_n_gate[0])?;
                let gate = ops::sigmoid(&hid_n_gate[1])?;
                match prev_hidden {
                    Some(prev_h) => {
                        out = lerp(prev_h, &hidden, &gate)?;
                    }
                    None => {
                        out = hidden.mul(&gate)?;
                    }
                }
            }
            sq_len => {
                let mut log_coeffs = (-1.0 * softplus(&hid_n_gate[1], 1.0, 20.0)?)?;
                let log_z = (-1.0 * softplus(&(-1.0 * &hid_n_gate[1])?, 1.0, 20.0)?)?;
                let log_tilde_h = log_g(&hid_n_gate[0])?;
                let mut log_values = log_z.add(&log_tilde_h)?;

                if prev_hidden.is_some() {
                    log_values = Tensor::cat(&[prev_hidden.unwrap().log()?, log_values], 1)?;
                    log_coeffs = log_coeffs.pad_with_zeros(D::Minus1, 1, 0)?;
                }
                let hscan_out = heinsen_associative_scan_log(&log_coeffs, &log_values)?;
                let dim1_size = hscan_out.dims()[1];
                out = hscan_out.i((.., dim1_size - sq_len..))?;
            }
        }

        let nxt_prv_hid = out.i((.., out.dims()[1] - 1..))?;

        if self.exp_factor != 1.0 {
            out = self.to_out.forward(&out)?;
        }

        let next_prev_hidden = match return_next_prev_hidden {
            false => None,
            true => Some(nxt_prv_hid),
        };

        Ok((out, next_prev_hidden))
    }
}

fn main() {
    println!("Hello, world!");
}
