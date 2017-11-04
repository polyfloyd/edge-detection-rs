Edge Detection
==============
[![Build Status](https://travis-ci.org/polyfloyd/edge-detection-rs.svg)](https://travis-ci.org/polyfloyd/edge-detection-rs)
[![](https://img.shields.io/crates/v/edge-detection.svg)](https://crates.io/crates/edge-detection)
[![Documentation](https://docs.rs/edge-detection/badge.svg)](https://docs.rs/edge-detection/)

An implementation of the Canny edge detection algorithm in Rust. The base for
many computer vision applications.

```rust
extern crate edge_detection;
extern crate image;

let source_image = image::open("testdata/line-simple.png")
    .expect("failed to read image")
    .to_luma();
let detection = edge_detection::canny(
    source_image,
    1.2,  // sigma
    0.2,  // strong threshold
    0.01, // weak threshold
);
```

![Circle](media/demo-circle.png)

![Peppers](media/demo-peppers.png)
