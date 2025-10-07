use pyo3::prelude::*;
use std::{collections::HashMap};

pub mod components;
pub use components::*;



fn initialization(original: HashMap<Vec<Token>, i32>) -> (Vocab, Merges, SequencePool, PairCounter, PairPos) {
    let vocab: Vocab = Vocab::new(256);
    let merges: Merges = Merges::new();
    let mut seq_pool = SequencePool::new();
    let mut pair_counter = PairCounter::new();
    let mut pair_pos = PairPos::new();
    // Scan original to fill seq_pool, pair_counter, pair_pos
    for (seq, count) in original.iter() {
        let seq_id = seq_pool.intern_and_update(seq.clone(), *count as i64);
        for window in seq.windows(2) {
            let pair = (window[0], window[1]);
            pair_counter.update(pair, *count as i64);
            pair_pos.add_position(pair, seq_id);
        }
    }
    //Return all these structures
    (vocab, merges, seq_pool, pair_counter, pair_pos)
}

fn perform_merge(
    best_pair: TokenPair,
    new_token: Token,
    seq_pool: &mut SequencePool,
    pair_counter: &mut PairCounter,
    pair_pos: &mut PairPos,
) {
    let positions = match pair_pos.get_positions(best_pair) {
        Some(pos) => pos,
        None => return, 
    };

    for &seq_id in &positions {
        let (seq, count) = 
        if let Some(seq_data) = seq_pool.get_data_by_id(seq_id) {
            (seq_data.seq.clone(), seq_data.count)
        } else {
            continue;
        };

        let mut new_seq = Vec::new();
        let mut i = 0;
        while i < seq.len() {
            if i < seq.len() - 1 && (seq[i], seq[i + 1]) == best_pair {
                new_seq.push(new_token);
                i += 2; // Skip the next token as it's part of the merged pair
            } else {
                new_seq.push(seq[i]);
                i += 1;
            }
        }
        // Update the sequence in the pool
        let new_seq_id = seq_pool.intern_and_update(new_seq.clone(), count);
        let old_seq_id = seq_pool.intern_and_update(seq.clone(), -count);
        // Update pair_counter and pair_pos for the new sequence
        for window in new_seq.windows(2) {
            let pair = (window[0], window[1]);
            pair_counter.update(pair, count);
            pair_pos.add_position(pair, new_seq_id);
        }
        // Update pair_counter and pair_pos for the old sequence
        for window in seq.windows(2) {
            let pair = (window[0], window[1]);
            pair_counter.update(pair, -count);
            pair_pos.erase_position(pair, old_seq_id);
        }
    }
}

#[pyfunction]
fn train_bpe(original: HashMap<Vec<Token>, i32>, num_merges: i32) -> (HashMap<Token, Vec<Byte>>, Vec<(Vec<Byte>, Vec<Byte>)>) {
    let (mut vocab, mut merges, mut seq_pool, mut pair_counter, mut pair_pos) = initialization(original);
    for new_token in 256..(256 + num_merges as Token) {
        if let Some((best_pair, _count)) = pair_counter.top_pair(&vocab) {
            merges.add_merge(best_pair, &vocab);
            perform_merge(best_pair, new_token, &mut seq_pool, &mut pair_counter, &mut pair_pos);
            vocab.insert(new_token, best_pair);
        } else {
            break; 
        }
    }
    (vocab.extract(), merges.extract())
}



/// A Python module implemented in Rust.
#[pymodule]
fn accelerator_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}
