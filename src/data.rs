use candle_core::{Device, Result, Tensor};
use flate2::bufread::GzDecoder;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::usize;

fn extract_data(filepath: &str) -> Vec<u8> {
    let file = File::open(filepath).unwrap();
    let bufreader = BufReader::new(file);
    let mut gz = GzDecoder::new(bufreader);
    let mut bytes = Vec::new();
    gz.read_to_end(&mut bytes).unwrap();
    bytes
}

pub struct Dataset {
    pub train_data: Vec<u8>,
    pub val_data: Vec<u8>,
    pub seq_len: usize,
    pub rng: ThreadRng,
}

impl Dataset {
    pub fn new(filepath: &str, seq_len: usize) -> Self {
        let data = extract_data(filepath);
        let (train_data, val_data) = data.split_at(90e6 as usize);
        let rng = rand::thread_rng();
        Self {
            train_data: train_data.to_vec(),
            val_data: val_data.to_vec(),
            seq_len,
            rng,
        }
    }

    pub fn get_batch(&mut self, mode: &str, batch_size: usize) -> Result<Tensor> {
        let mut batch: Vec<u8> = Vec::new();
        for _ in 0..batch_size {
            let mut sample = if mode == "train" {
                self.get_train_sample()
            } else {
                self.get_val_sample()
            };
            batch.append(&mut sample);
        }
        Tensor::from_vec(batch, (batch_size, self.seq_len), &Device::Cpu)
    }

    fn get_train_sample(&mut self) -> Vec<u8> {
        let rnd_idx = self
            .rng
            .gen_range(0..(self.train_data.len() - self.seq_len));
        self.train_data[rnd_idx..rnd_idx + self.seq_len].to_vec()
    }

    fn get_val_sample(&mut self) -> Vec<u8> {
        let rnd_idx = self.rng.gen_range(0..(self.val_data.len() - self.seq_len));
        self.val_data[rnd_idx..rnd_idx + self.seq_len].to_vec()
    }
}
