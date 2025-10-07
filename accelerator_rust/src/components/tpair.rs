use std::cmp::Ordering;
use std::collections::{HashMap,BTreeSet, HashSet};

use super::output::{Token, TokenPair, Vocab};
use super::sequence::SeqId;

type Pair = u64;
#[inline] fn pack_pair(a: Token, b: Token) -> Pair {
    ((a as Pair) << 32) | (b as Pair)
}
#[inline] fn unpack_pair(p: Pair) -> TokenPair {
    ((p >> 32) as Token, (p & 0xFFFFFFFF) as Token)
}


pub struct PairCounter{
    counts: HashMap<Pair, i64>,
    buckets: HashMap<i64, HashSet<Pair>>,
    nonzero_counts: BTreeSet<i64>,
}

impl PairCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            buckets: HashMap::new(),
            nonzero_counts: BTreeSet::new(),
        }
    }

    pub fn update(&mut self, pair: TokenPair, delta: i64) {
        if delta == 0 {
            return;
        }
        let p = pack_pair(pair.0, pair.1);
        let old_count = self.counts.get(&p).copied().unwrap_or(0);
        let new_count = old_count + delta;

        if old_count > 0 {
            if let Some(bucket) = self.buckets.get_mut(&old_count) {
                bucket.remove(&p);
                if bucket.is_empty() {
                    self.buckets.remove(&old_count);
                    self.nonzero_counts.remove(&old_count);
                }
            }
        }

        if new_count <= 0 {
            self.counts.remove(&p);
            return;
        }

        let bucket = self.buckets.entry(new_count).or_insert_with(HashSet::new);
        let new_bucket = bucket.is_empty();
        bucket.insert(p);
        if new_bucket {
            self.nonzero_counts.insert(new_count);
        }
        self.counts.insert(p, new_count);
    }

    pub fn top_pair(&self, vocab: &Vocab) -> Option<(TokenPair, i64)> {
        let &max_count = self.nonzero_counts.iter().next_back()?; // 最大计数
        let bucket = self.buckets.get(&max_count)?;
        let &best = bucket
            .iter()
            .max_by(|x, y| Self::lex_cmp(vocab, **x, **y))?;
        Some((unpack_pair(best), max_count))
    }

    #[inline]
    fn lex_cmp(vocab: &Vocab, p1: Pair, p2: Pair) -> Ordering {
        let (a1, b1) = unpack_pair(p1);
        let (a2, b2) = unpack_pair(p2);

        let a1s = vocab.get(a1).map(Vec::as_slice).unwrap_or(&[]);
        let a2s = vocab.get(a2).map(Vec::as_slice).unwrap_or(&[]);
        match a1s.cmp(a2s) {
            Ordering::Equal => {
                let b1s = vocab.get(b1).map(Vec::as_slice).unwrap_or(&[]);
                let b2s = vocab.get(b2).map(Vec::as_slice).unwrap_or(&[]);
                b1s.cmp(b2s)
            }
            other => other,
        }
    }
}


pub struct PairPos {
    pub positions: HashMap<Pair, HashMap<SeqId, i64>>,
}

impl PairPos {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
        }
    }

    pub fn add_position(&mut self, pair: TokenPair, seq_id: SeqId) {
        let p = pack_pair(pair.0, pair.1);
        let seq_counts = self.positions.entry(p).or_insert_with(HashMap::new);
        *seq_counts.entry(seq_id).or_insert(0) += 1;
    }

    pub fn get_positions(&self, pair: TokenPair) -> Option<Vec<SeqId>> {
        let p = pack_pair(pair.0, pair.1);
        self.positions.get(&p).map(|seq_counts| {
            seq_counts.keys().copied().collect()
        })
    }


    pub fn erase_position(&mut self, pair: TokenPair, seq_id: SeqId) {
        let p = pack_pair(pair.0, pair.1);

        if let Some(seq_counts) = self.positions.get_mut(&p) {
            seq_counts.remove(&seq_id);
            if let Some(count) = seq_counts.get_mut(&seq_id) {
                *count -= 1;
                
                if *count == 0 {
                    seq_counts.remove(&seq_id);
                }
            }
            
            if seq_counts.is_empty() {
                self.positions.remove(&p);
            }
        }
    }
}