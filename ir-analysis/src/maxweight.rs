#![allow(clippy::needless_range_loop)]
use pathfinding::kuhn_munkres::{kuhn_munkres, Weights};

struct Matching(Vec<Vec<i32>>);

pub fn matching(matrix: Vec<Vec<i32>>) -> Vec<usize> {
    let matching = Matching(matrix);
    let (_, match_) = kuhn_munkres(&matching);
    match_
}

impl Weights<i32> for Matching {
    fn rows(&self) -> usize {
        self.0.len()
    }

    fn columns(&self) -> usize {
        self.0[0].len()
    }

    fn at(&self, row: usize, col: usize) -> i32 {
        self.0[row][col]
    }

    fn neg(&self) -> Self
    where
        Self: Sized,
        i32: pathfinding::num_traits::Signed,
    {
        let mut neg = vec![vec![0; self.0[0].len()]; self.0.len()];
        for i in 0..self.0.len() {
            for j in 0..self.0[0].len() {
                neg[i][j] = -self.0[i][j];
            }
        }
        Matching(neg)
    }
}
