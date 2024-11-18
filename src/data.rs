use anyhow::Error as E;
use flate2::bufread::GzDecoder;
use hf_hub::api::sync::Api;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::usize;
use tokenizers::tokenizer::{Encoding, Tokenizer};
use tokenizers::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};

fn extract_data(filepath: &str) -> Vec<u8> {
    let file = File::open(filepath).unwrap();
    let bufreader = BufReader::new(file);
    let mut gz = GzDecoder::new(bufreader);
    let mut bytes = Vec::new();
    gz.read_to_end(&mut bytes).unwrap();
    bytes
}

fn get_tokenizer(tokenizer_file: Option<&str>) -> Result<Tokenizer, E> {
    let api = Api::new()?;
    let tokenizer_filename = match tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => api.model("Xenova/gpt2".to_string()).get("tokenizer.json")?,
    };
    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

pub struct Dataset {
    pub train_data: Vec<u8>,
    pub val_data: Vec<u8>,
    pub tokenizer: Tokenizer,
    pub padding_params: PaddingParams,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rng: ThreadRng,
}

impl Dataset {
    pub fn new(filepath: &str, tokenizer_file: Option<&str>, seq_len: usize) -> Self {
        let data = extract_data(filepath);
        let (train_data, val_data) = data.split_at(90e6 as usize);
        let tokenizer = get_tokenizer(tokenizer_file).unwrap();
        let padding_params = match tokenizer.get_padding() {
            Some(pad_params) => pad_params.to_owned(),
            None => PaddingParams {
                strategy: PaddingStrategy::Fixed(seq_len),
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: 0,
                pad_type_id: 0,
                pad_token: String::from("[PAD]"),
            },
        };
        let vocab_size = tokenizer.get_vocab_size(false);
        let rng = rand::thread_rng();
        Self {
            train_data: train_data.to_vec(),
            val_data: val_data.to_vec(),
            tokenizer,
            padding_params,
            vocab_size,
            seq_len,
            rng,
        }
    }

    pub fn get_batch(&mut self, mode: &str, batch_size: usize) -> Result<Vec<Encoding>, E> {
        let mut batch: Vec<Encoding> = Vec::new();
        for _ in 0..batch_size {
            let sample = if mode == "train" {
                self.get_train_sample()
            } else {
                self.get_val_sample()
            };
            batch.push(sample);
        }
        pad_encodings(&mut batch, &self.padding_params).unwrap();
        Ok(batch)
    }

    fn find_nearest_ascii(&self, mut idx: usize, max_idx: usize, mode: &str) -> usize {
        loop {
            let is_ascii = if mode == "train" {
                self.train_data[idx].is_ascii()
            } else {
                self.val_data[idx].is_ascii()
            };
            if is_ascii || (idx > max_idx) {
                break;
            } else {
                idx += 1;
            }
        }
        idx
    }

    fn get_train_sample(&mut self) -> Encoding {
        let max_idx = self.train_data.len() - self.seq_len - 1;
        let mut rnd_start = self.rng.gen_range(0..(max_idx + 1));
        rnd_start = self.find_nearest_ascii(rnd_start, max_idx, "train");
        let mut end = rnd_start + self.seq_len;
        end = self.find_nearest_ascii(end, max_idx, "train");
        let input = &self.train_data[rnd_start..end];
        let text_input = match String::from_utf8(input.to_vec()) {
            Ok(txt) => txt,
            Err(_) => String::from("Failed to get string from source text"),
        };
        self.tokenizer.encode_fast(text_input, false).unwrap()
    }

    fn get_val_sample(&mut self) -> Encoding {
        let max_idx = self.val_data.len() - self.seq_len - 1;
        let mut rnd_start = self.rng.gen_range(0..(self.val_data.len() - self.seq_len));
        rnd_start = self.find_nearest_ascii(rnd_start, max_idx, "val");
        let mut end = rnd_start + self.seq_len;
        end = self.find_nearest_ascii(end, max_idx, "val");
        let input = &self.val_data[rnd_start..end];
        let text_input = match String::from_utf8(input.to_vec()) {
            Ok(txt) => txt,
            Err(_) => String::from("Failed to get string from source text"),
        };
        self.tokenizer.encode_fast(text_input, false).unwrap()
    }
}
