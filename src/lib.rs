#![cfg_attr(all(test, feature = "unstable"), feature(test))]

extern crate image;
extern crate rayon;

mod edge;

pub use edge::*;
