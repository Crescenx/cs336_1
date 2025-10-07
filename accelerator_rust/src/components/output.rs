
use std::collections::HashMap;

pub type Token = u32;
pub type TokenPair = (Token, Token);
pub type Byte = u8;

pub struct Vocab {
    pub map: HashMap<Token, Vec<Byte>>,
}

pub struct Merges {
    pub list: Vec<(Vec<Byte>, Vec<Byte>)>,
}

impl Vocab {
    pub fn new(initial_size: usize) -> Self {
        let mut map = HashMap::with_capacity(initial_size);
        for i in 0..256 {
            map.insert(i as Token, vec![i as Byte]);
        }
        Self { map }
    }
    

    pub fn get(&self, token: Token) -> Option<&Vec<Byte>> {
        self.map.get(&token)
    }

    pub fn insert(&mut self, token: Token, pair: TokenPair) {
        let (p1, p2) = pair;
        let decode_p1 = self.map.get(&p1).cloned().unwrap_or_else(|| vec![p1 as Byte]);
        let decode_p2 = self.map.get(&p2).cloned().unwrap_or_else(|| vec![p2 as Byte]);
        self.map.insert(token, [decode_p1, decode_p2].concat());
    }

    pub fn extract(self) -> HashMap<Token, Vec<Byte>> {
        self.map
    }
}

impl Merges {
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    pub fn add_merge(&mut self, pair: TokenPair, vocab: &Vocab) {
        let (p1, p2) = pair;
        let decode_p1 = vocab.get(p1).cloned().unwrap_or_else(|| vec![p1 as Byte]);
        let decode_p2 = vocab.get(p2).cloned().unwrap_or_else(|| vec![p2 as Byte]);
        self.list.push((decode_p1, decode_p2));
    }

    pub fn extract(self) -> Vec<(Vec<Byte>, Vec<Byte>)> {
        self.list
    }
}