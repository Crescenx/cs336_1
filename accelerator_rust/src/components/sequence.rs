
use std::collections::HashMap;
// Unique identifier for each sequence
pub type SeqId = u32;
use super::output::{Token};

#[derive(Debug, Clone)]
pub struct SequenceData {
    pub seq: Vec<Token>,
    pub count: i64, 
}


pub struct SequencePool {
    intern_pool: HashMap<Vec<Token>, SeqId>,
    data_storage: Vec<SequenceData>,
}

impl SequencePool {
    pub fn new() -> Self {
        Self {
            intern_pool: HashMap::new(),
            data_storage: Vec::new(),
        }
    }

    pub fn intern_and_update(&mut self, seq: Vec<Token>, delta: i64) -> SeqId {
        if let Some(&id) = self.intern_pool.get(&seq) {
            self.data_storage[id as usize].count += delta;
            id
        } else {
            let new_id = self.data_storage.len() as SeqId;
            let new_data = SequenceData {
                seq: seq.clone(), 
                count: delta,
            };
            self.data_storage.push(new_data);
            self.intern_pool.insert(seq, new_id);
            new_id
        }
    }

    pub fn get_data_by_id(&self, id: SeqId) -> Option<&SequenceData> {
        self.data_storage.get(id as usize)
    }

    pub fn get_id_by_seq(&self, seq: &[Token]) -> Option<SeqId> {
        self.intern_pool.get(seq).copied()
    }

    pub fn get_data_by_seq(&self, seq: &[Token]) -> Option<(SeqId, &SequenceData)> {
        self.intern_pool.get(seq).and_then(|&id| {
            self.get_data_by_id(id).map(|data| (id, data))
        })
    }

    pub fn len(&self) -> usize {
        self.data_storage.len()
    }
}

