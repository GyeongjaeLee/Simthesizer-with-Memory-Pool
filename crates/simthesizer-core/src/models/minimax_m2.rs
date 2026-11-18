use anyhow::{ensure, Result};
use serde::Deserialize;

use crate::model::Model;
use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDtype {
    #[default]
    Bf16,
    Fp8,
}

/// MiniMax-M2 with block-FP8 linear weights; FP32 router and BF16
/// norms/embedding/LM head stay unquantized, as in the checkpoint. No MTP.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxM2Model {
    pub hidden_size: u64,
    /// Per-expert FFN width.
    pub intermediate_size: u64,
    pub num_attention_heads: u64,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u64,
    pub num_local_experts: u64,
    pub num_experts_per_tok: u32,
    pub vocab_size: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    #[serde(default)]
    pub kv_dtype: KvDtype,
    #[serde(skip)]
    weight_bytes: u64,
    #[serde(skip)]
    kv_bytes_per_token: u64,
    #[serde(skip)]
    pre: Vec<Layer>,
    #[serde(skip)]
    block: Vec<Layer>,
    #[serde(skip)]
    post: Vec<Layer>,
}

crate::register_model!(MiniMaxM2Model, "minimax_m2");
impl Module for MiniMaxM2Model {}

const FP8_BLOCK: u64 = 128;

fn gemm(name: &str, m: u64, n: u64) -> Layer {
    Layer::Gemm {
        name: name.into(),
        m,
        n,
    }
}

fn norm(name: &str, hidden_size: u64) -> Layer {
    Layer::LayerNorm {
        name: name.into(),
        hidden_size,
    }
}

fn all_reduce(name: &str, bytes_per_token: u64) -> Layer {
    Layer::AllReduce {
        name: name.into(),
        bytes_per_token,
    }
}

impl Model for MiniMaxM2Model {
    /// Attention heads, experts and vocabulary are TP-sharded.
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()> {
        ensure!(
            parallel.tp > 0 && parallel.ep == 1 && parallel.pp == 1,
            "MiniMax-M2 supports TP with EP1/PP1"
        );
        ensure!(
            [
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.num_local_experts,
                self.vocab_size,
                self.head_dim,
            ]
            .into_iter()
            .all(|v| v > 0)
                && self.num_hidden_layers > 0
                && self.num_experts_per_tok > 0
                && u64::from(self.num_experts_per_tok) <= self.num_local_experts,
            "MiniMax-M2 dimensions must be positive with top-k within the experts"
        );
        let tp = u64::from(parallel.tp);
        ensure!(
            self.num_attention_heads.is_multiple_of(tp)
                && self.num_key_value_heads.is_multiple_of(tp)
                && self.intermediate_size.is_multiple_of(tp)
                && self.vocab_size.is_multiple_of(tp),
            "MiniMax-M2 heads, FFN and vocabulary must divide TP"
        );
        ensure!(
            self.rotary_dim <= self.head_dim && self.rotary_dim.is_multiple_of(2),
            "MiniMax-M2 rotary_dim must be even and at most head_dim"
        );
        let q_heads = self.num_attention_heads / tp;
        let kv_heads = self.num_key_value_heads / tp;
        let q = q_heads * self.head_dim;
        let kv = kv_heads * self.head_dim;
        let ffn = self.intermediate_size / tp;
        let experts = self.num_local_experts;
        ensure!(
            [self.hidden_size, q, kv, ffn]
                .into_iter()
                .all(|v| v.is_multiple_of(FP8_BLOCK)),
            "MiniMax-M2 local FP8 matrices must align to 128x128 scale blocks"
        );
        let fp8_per_layer = self.hidden_size * (q + 2 * kv)
            + q * self.hidden_size
            + experts * 3 * self.hidden_size * ffn;
        let fp32_scales_per_layer = fp8_per_layer / (FP8_BLOCK * FP8_BLOCK) * 4;
        let fp32_router_per_layer = (self.hidden_size + 1) * self.num_local_experts * 4;
        let bf16_norms_per_layer = (2 * self.hidden_size + q + kv) * 2;
        let layers = u64::from(self.num_hidden_layers);
        self.weight_bytes = layers
            * (fp8_per_layer
                + fp32_scales_per_layer
                + fp32_router_per_layer
                + bf16_norms_per_layer)
            + self.hidden_size * 2
            + 2 * (self.vocab_size / tp) * self.hidden_size * 2;
        let kv_element_bytes = match self.kv_dtype {
            KvDtype::Bf16 => 2,
            KvDtype::Fp8 => 1,
        };
        self.kv_bytes_per_token = 2 * kv * kv_element_bytes * layers;

        self.pre = vec![Layer::Embedding {
            name: "embedding".into(),
            num_embeddings: self.vocab_size / tp,
            embedding_dim: self.hidden_size,
        }];
        self.block = vec![
            norm("input_layernorm", self.hidden_size),
            gemm("qkv_projection", self.hidden_size, q + 2 * kv),
            norm("qk_norm", q + kv),
            Layer::Rope {
                name: "rope".into(),
                head_dim: self.rotary_dim,
                num_heads: q_heads + kv_heads,
            },
            Layer::Attention {
                name: "attention".into(),
                num_q_heads: q_heads,
                num_kv_heads: kv_heads,
                head_dim: self.head_dim,
            },
            gemm("o_projection", q, self.hidden_size),
            all_reduce("attention_all_reduce", self.hidden_size * 2),
            norm("post_attention_layernorm", self.hidden_size),
            gemm("router", self.hidden_size, self.num_local_experts),
            Layer::Moe {
                name: "moe".into(),
                num_experts: self.num_local_experts,
                num_experts_per_token: self.num_experts_per_tok,
                num_shared_experts_per_token: 0,
                dispatch_bytes_per_token: 0,
                combine_bytes_per_token: 0,
            },
            all_reduce("moe_all_reduce", self.hidden_size * 2),
        ];
        self.post = vec![
            norm("final_layernorm", self.hidden_size),
            gemm("lm_head", self.hidden_size, self.vocab_size / tp),
            Layer::Sampler {
                name: "sampler".into(),
                vocab_size: self.vocab_size,
            },
        ];
        Ok(())
    }

    fn weight_bytes_per_device(&self) -> u64 {
        self.weight_bytes
    }

    fn kv_bytes_per_token_per_device(&self) -> u64 {
        self.kv_bytes_per_token
    }

    fn pre_block_layers(&self) -> &[Layer] {
        &self.pre
    }

    fn block_layers(&self) -> &[Layer] {
        &self.block
    }

    fn post_block_layers(&self) -> &[Layer] {
        &self.post
    }

    fn num_blocks(&self) -> u32 {
        self.num_hidden_layers
    }
}
