#![cfg_attr(all(test, feature = "unstable"), feature(test))]

extern crate image;

mod edge;

pub use edge::*;
