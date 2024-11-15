# minGRU
Implementation of [minGRU](https://arxiv.org/abs/2410.01201) in Rust using the [candle](https://github.com/huggingface/candle) framework.

This is a port of [minGRU-pytorch](https://github.com/lucidrains/minGRU-pytorch) as a small experiment to get more familiar with `candle`.

Not finished yet, still work in progress.

## Usage
Download [enwiki8.gz](https://github.com/lucidrains/minGRU-pytorch/blob/main/data/enwik8.gz) and put it in `data/` folder inside this repo. Then:

```bash
cargo run -r
```
It will launch training with character level tokeninzation.

## Citations

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
