#![cfg_attr(all(test, feature = "unstable"), feature(test))]
#![warn(missing_docs)]

//! An implementation of the Canny edge detection algorithm in Rust. The base for
//! many computer vision applications.
//!
//! # Finding the edges in an image
//!
//! ```
//! extern crate edge_detection;
//! extern crate image;
//!
//! let source_image = image::open("testdata/line-simple.png")
//!     .expect("failed to read image")
//!     .to_luma();
//! let detection = edge_detection::canny(
//!     source_image,
//!     1.2,  // sigma
//!     0.2,  // strong threshold
//!     0.01, // weak threshold
//! );
//! ```
//!
//! See the `canny` function for details on what each parameter means.

extern crate image;
extern crate rayon;

mod edge;

pub use edge::*;
