use std::*;
use std::f32::consts::*;
use image;
use rayon::prelude::*;


#[derive(Clone)]
pub struct Detection {
    pub edges: Vec<Vec<Edge>>,
}

impl Detection {
    pub fn width(&self) -> usize {
        self.edges.len()
    }

    pub fn height(&self) -> usize {
        self.edges.first().unwrap().len()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Edge {
    pub vec_x: f32,
    pub vec_y: f32,
    pub magnitude: f32,
}

impl Edge {
    fn new(vec_x: f32, vec_y: f32) -> Edge {
        let vec_x = FRAC_1_SQRT_2 * if vec_x > 1.0 {
            1.0
        } else if vec_x < -1.0 {
            -1.0
        } else {
            vec_x
        };
        let vec_y = FRAC_1_SQRT_2 * if vec_y > 1.0 {
            1.0
        } else if vec_y < -1.0 {
            -1.0
        } else {
            vec_y
        };
        let magnitude = (vec_x.powi(2) + vec_y.powi(2)).sqrt();
        assert!(0.0 <= magnitude && magnitude <= 1.0);
        let frac_1_mag = if magnitude != 0.0 {
            1.0 / magnitude
        } else {
            1.0
        };
        Edge {
            vec_x: vec_x * frac_1_mag,
            vec_y: vec_y * frac_1_mag,
            magnitude,
        }
    }

    fn theta(&self) -> f32 {
        self.vec_y.atan2(self.vec_x)
    }
}

pub fn canny<T: Into<image::GrayImage>>(image: T, sigma: f32, strong_threshold: f32, weak_threshold: f32) -> Detection {
    let edges = detect_edges(&image.into(), sigma);
    let edges = minmax_suppression(&edges);
    let edges = hysteresis(&edges, strong_threshold, weak_threshold);
    Detection { edges }
}

/// Calculates a 2nd order 2D gaussian derivative with size sigma.
fn filter_kernel(sigma: f32) -> Vec<Vec<(f32, f32)>> {
    let size = (sigma * 10.0).round() as usize;
    (0..size).map(|x| {
        (0..size).map(|y| {
            let (xf, yf) = (x as f32 - size as f32 / 2.0, y as f32 - size as f32 / 2.0);
            let g = E.powf(-(xf.powi(2) + yf.powi(2)) / (2. * sigma.powi(2))) / (2.0 * sigma.powi(2));
            (xf * g, yf * g)
        })
        .collect()
    })
    .collect()
}

fn neighbour_pos_delta(theta: f32) -> (i32, i32) {
    let neighbours = [
        (1, 0),   // middle right
        (1, 1),   // bottom right
        (0, 1),   // center bottom
        (-1, 1),  // bottom left
        (-1, 0),  // middle left
        (-1, -1), // top left
        (0, -1),  // center top
        (1, -1),  // top right
    ];
    let n = ((theta + PI * 2.0) % (2.0 * PI)) / (2.0 * PI);
    let i = (n * 8.0 + 0.001).floor().min(7.0);
    assert!(0.0 <= i && i < 8.0);
    neighbours[i as usize]
}

/// Computes the edges in an image using the Canny Method.
///
/// `sigma` determines the radius of the Gaussian kernel.
fn detect_edges(image: &image::GrayImage, sigma: f32) -> Vec<Vec<Edge>> {
    let (width, height) = (image.width() as i32, image.height() as i32);
    let kernel = filter_kernel(sigma);
    (0..width).into_par_iter().map(|ix| {
        (0..height).into_par_iter().map(|iy| {
            let ks = kernel.len() as i32;
            let (sum_x, sum_y) = kernel.iter()
                .zip(-ks / 2..ks / 2)
                .flat_map(|(col, kx)| {
                    col.iter()
                        .zip(-ks / 2..ks / 2)
                        .map(move |(k, ky)| {
                            // Clamp x and y within the image bounds so no non-existing borders are be
                            // detected based on some background color outside image bounds.
                            let x = (ix + kx).min(width - 1).max(0);
                            let y = (iy + ky).min(height - 1).max(0);
                            let pix = image.get_pixel(x as u32, y as u32).data[0] as f32 / 255.0;
                            (pix * k.0, pix * k.1)
                        })
                })
                .fold((0.0, 0.0), |accum, pix| (accum.0 + pix.0, accum.1 + pix.1));
            Edge::new(sum_x, sum_y)
        })
        .collect()
    })
    .collect()
}

/// Narrows the width of detected edges down to a single pixel.
fn minmax_suppression(edges: &Vec<Vec<Edge>>) -> Vec<Vec<Edge>> {
    let (width, height) = (edges.len(), edges[0].len());
    (0..width).into_par_iter().map(|x| {
        (0..height).into_par_iter().map(|y| {
            let edge = edges[x][y];
            if edge.magnitude < 0.0001 {
                // Skip distance computation for non-edges.
                return Edge::new(0.0, 0.0);
            }

            let distances: Vec<i32> = [1.0, -1.0].into_iter()
                .map(|side| {
                    // A vector confined to a box instead of a radius.
                    // The magnitude ranges from 1 to sqrt(2).
                    let box_vec = (
                        (1.0 / edge.vec_x * side).min(1.0).max(-1.0),
                        (1.0 / edge.vec_y * side).min(1.0).max(-1.0),
                    );
                    assert!({
                        let m = (box_vec.0.powi(2) + box_vec.1.powi(2)).sqrt();
                        1.0 <= m && m <= SQRT_2
                    });

                    // Truncating the edge magnitudes helps mitigate rounding errors for thick edges.
                    let truncate = |f: f32| (f * 1e5).round() * 1e-6;

                    let mut seek_pos = (x as f32, y as f32);
                    let mut seek_magnitude = truncate(edge.magnitude);
                    let mut distance = 0;
                    loop {
                        seek_pos.0 += box_vec.0;
                        seek_pos.1 += box_vec.1;
                        let (nb_a, nb_b, n) = if (seek_pos.0 + 0.5).fract() < (seek_pos.1 + 0.5).fract() {
                            // X is closest to a point.
                            let x = (seek_pos.0.round() as usize).min(width - 1);
                            let y1 = seek_pos.1.floor().max(0.0) as usize;
                            let y2 = (seek_pos.1.ceil() as usize).min(height - 1);
                            let n = seek_pos.1.fract();
                            (
                                edges.get(x).and_then(|col| col.get(y1)),
                                edges.get(x).and_then(|col| col.get(y2)),
                                n,
                            )
                        } else {
                            // Y is closest to a point.
                            let y = (seek_pos.1.round() as usize).min(height - 1);
                            let x1 = seek_pos.0.floor().max(0.0) as usize;
                            let x2 = (seek_pos.0.ceil() as usize).min(width - 1);
                            let n = seek_pos.0.fract();
                            (
                                edges.get(x1).and_then(|col| col.get(y)),
                                edges.get(x2).and_then(|col| col.get(y)),
                                n,
                            )
                        };
                        assert!(n >= 0.0);

                        if let (Some(nb_a), Some(nb_b)) = (nb_a, nb_b) {
                            let tr_edge_mag = truncate(edge.magnitude);
                            let interpolated_magnitude = truncate(nb_a.magnitude * (1.0 - n) + nb_b.magnitude * n);
                            if seek_magnitude > tr_edge_mag && interpolated_magnitude < seek_magnitude {
                                break;
                            } else if interpolated_magnitude < tr_edge_mag {
                                break;
                            } else {
                                seek_magnitude = interpolated_magnitude;
                            }
                        } else {
                            break;
                        }
                        distance += 1;
                    }
                    distance
                })
                .collect();

            // Equal distances denote the middle of the edge.
            // A deviation of 1 is allowed for edges over two equal pixels, in which case, the
            // outer edge (near the dark side) is preferred.
            let is_apex =
                // The distances are equal, the edge's width is odd, making the apex lie on a
                // single pixel.
                distances[0] == distances[1]
                // There is a difference of 1, the edge's width is even, spreading the apex over
                // two pixels. This is a special case to handle edges that run along either the X- or X-axis.
                || (distances[0] - distances[1] == 1) && ((1.0 - edge.vec_x.abs()).abs() < 0.001 || (1.0 - edge.vec_y.abs()).abs() < 0.001);
            if is_apex {
                edge
            } else {
                Edge::new(0.0, 0.0)
            }
        })
        .collect()
    })
    .collect()
}

fn hysteresis(edges: &Vec<Vec<Edge>>, strong_threshold: f32, weak_threshold: f32) -> Vec<Vec<Edge>> {
    assert!(0.0 <= strong_threshold && strong_threshold < 1.0);
    assert!(0.0 <= weak_threshold && weak_threshold < 1.0);
    assert!(weak_threshold < strong_threshold);
    let (width, height) = (edges.len(), edges.first().unwrap().len());
    let mut edges_out: Vec<Vec<Edge>> = vec![vec![Edge::new(0.0, 0.0); height]; width];
    for x in 0..width {
        for y in 0..height {
            let edge = edges[x][y];
            if edge.magnitude >= strong_threshold && edges_out[x][y].magnitude != 1.0 {
                // Start following in both directions.
                let mut stack = vec![(x, y)];
                while let Some(top) = stack.pop() {
                    edges_out[top.0][top.1] = edges[top.0][top.1];
                    edges_out[top.0][top.1].magnitude = 1.0;

                    for invert in [0.0, PI].into_iter() {
                        let (nb_pos, nb_magnitude) = [-FRAC_PI_4, 0.0, FRAC_PI_4].into_iter()
                            .map(|bearing| {
                                neighbour_pos_delta(edge.theta() + invert + bearing)
                            })
                            .filter_map(|(nb_dx, nb_dy)| {
                                let nb_x = x as i32 + nb_dx;
                                let nb_y = y as i32 + nb_dy;
                                if nb_x < 0 || nb_x >= width as i32 || nb_y < 0 || nb_y >= height as i32 {
                                    return None;
                                }
                                let nb = (nb_x as usize, nb_y as usize);
                                Some((nb, edges[nb.0][nb.1].magnitude))
                            })
                            .fold(((0, 0), 0.0), |(max_pos, max_mag), (pos, mag)| {
                                if mag > max_mag {
                                    (pos, mag)
                                } else {
                                    (max_pos, max_mag)
                                }
                            });

                        if nb_magnitude >= weak_threshold && edges_out[nb_pos.0][nb_pos.1].magnitude != 1.0 {
                            stack.push(nb_pos);
                        }
                    }

                }
            }
        }
    }
    edges_out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_vectors_to_image(edges: &Vec<Vec<Edge>>) -> image::RgbImage {
        let (width, height) = (edges.len(), edges.first().unwrap().len());
        let mut image = image::RgbImage::from_pixel(width as u32, height as u32, image::Rgb{ data: [0, 0, 0] });
        for x in 0..width {
            for y in 0..height {
                let edge = edges[x][y];
                let pix = image.get_pixel_mut(x as u32, y as u32);
                match (edge.theta() + (PI * 2.0 + FRAC_PI_4)) % (PI * 2.0) {
                    t if t < FRAC_PI_2 => { // Right side
                        pix.data[0] = (edge.magnitude * 255.0) as u8;
                    },
                    t if t < PI => { // Bottom side
                        pix.data[1] = (edge.magnitude * 255.0) as u8;
                    },
                    t if t < PI + FRAC_PI_2 => { // Left side
                        pix.data[2] = (edge.magnitude * 255.0) as u8;
                    },
                    t if t < PI * 2.0 => { // Top side
                        pix.data[0] = (edge.magnitude * 255.0) as u8;
                        pix.data[1] = (edge.magnitude * 255.0) as u8;
                    },
                    _ => unreachable!(),
                };
            }
        }
        image
    }

    fn edges_to_image(edges: &Vec<Vec<Edge>>) -> image::GrayImage {
        let (width, height) = (edges.len(), edges.first().unwrap().len());
        let mut image = image::GrayImage::from_pixel(width as u32, height as u32, image::Luma{ data: [0] });
        for x in 0..width {
            for y in 0..height {
                let edge = edges[x][y];
                *image.get_pixel_mut(x as u32, y as u32) = image::Luma{ data: [(edge.magnitude * 255.0).round() as u8]};
            }
        }
        image
    }

    fn canny_output_stages<T: AsRef<str>>(path: T, sigma: f32, strong_threshold: f32, weak_threshold: f32) -> Detection {
        let path = path.as_ref();
        let image = image::open(path).unwrap();
        let edges = detect_edges(&image.to_luma(), sigma);
        edge_vectors_to_image(&edges).save(format!("{}.0-vectors.png", path)).unwrap();
        edges_to_image(&edges).save(format!("{}.1-edges.png", path)).unwrap();
        let edges = minmax_suppression(&edges);
        edges_to_image(&edges).save(format!("{}.2-minmax.png", path)).unwrap();
        let edges = hysteresis(&edges, strong_threshold, weak_threshold);
        edges_to_image(&edges).save(format!("{}.3-hysteresis.png", path)).unwrap();
        Detection { edges }
    }

    #[test]
    fn neighbour_pos_delta_from_theta() {
        let neighbours = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        for nb in neighbours.iter() {
            let d = neighbour_pos_delta(f32::atan2(nb.1 as f32, nb.0 as f32));
            assert_eq!(*nb, d);
        }
    }

    #[test]
    fn edge_new() {
        let e = Edge::new(1.0, 0.0);
        assert!(1.0 - 1e-6 < e.vec_x && e.vec_x < 1.0 + 1e-6);
        assert!(-1e-5 < e.vec_y && e.vec_y < 1e-6);

        let e = Edge::new(1.0, 1.0);
        assert!(FRAC_1_SQRT_2 - 1e-5 < e.vec_x && e.vec_x < FRAC_1_SQRT_2 + 1e-6);
        assert!(FRAC_1_SQRT_2 - 1e-5 < e.vec_y && e.vec_y < FRAC_1_SQRT_2 + 1e-6);
        assert!(1.0 - 1e-6 < e.magnitude && e.magnitude < 1.0 + 1e-6);
    }

    #[test]
    fn kernel_integral_in_bounds() {
        // The integral for the filter kernel should approximate 0.
        for sigma_i in 1..200 {
            let sigma = sigma_i as f32 / 10.0;
            let kernel = filter_kernel(sigma);
            let ksize = kernel.len();
            assert!(kernel.len() == kernel[0].len());
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for col in kernel {
                for (gx, gy) in col {
                    sum_x += gx;
                    sum_y += gy;
                }
            }
            println!("sum = ({}, {}), sigma = {}, kernel_size = {}", sum_x, sum_y, sigma, ksize);
            assert!(-0.0001 < sum_x && sum_x <= 0.0001);
            assert!(-0.0001 < sum_y && sum_y <= 0.0001);
        }
    }

    /// Tests whether a vertical line of 1px wide exists in the middle of the image.
    fn detect_vertical_line(detection: Detection) {
        // Find the line.
        let mut line_x = None;
        for x in 0..detection.width() {
            if detection.edges[x][detection.height() / 2].magnitude > 0.5 {
                if line_x.is_some() {
                    panic!("the line is thicker than 1px");
                }
                line_x = Some(x)
            }
        }
        let line_x = line_x.expect("no line detected");
        // The line should be at about the middle of the image.
        let middle = detection.width() / 2;
        assert!(middle - 1 <= line_x && line_x <= middle);
        // The line should be continuous.
        for y in 0..detection.height() {
            let edge = detection.edges[line_x][y];
            assert!(edge.magnitude == 1.0);
            // The direction of the line's surface normal should follow the X-axis.
            assert!(-0.05 < edge.theta() && edge.theta() <= 0.05);
        }
        // The line should be the only thing detected.
        for x in 0..detection.width() {
            if x == line_x {
                continue;
            }
            for y in 0..detection.height() {
                assert!(detection.edges[x][y].magnitude == 0.0);
            }
        }
    }

    #[test]
    fn detect_vertical_line_simple() {
        let d = canny_output_stages("testdata/line-simple.png", 1.2, 0.4, 0.05);
        detect_vertical_line(d);
    }

    #[test]
    fn detect_vertical_line_fuzzy() {
        let d = canny_output_stages("testdata/line-fuzzy.png", 2.0, 0.4, 0.05);
        detect_vertical_line(d);
    }
}

#[cfg(all(test, feature = "unstable"))]
mod benchmarks {
    extern crate test;
    use super::*;

    static IMG_PATH: &str = "testdata/circle.png";

    #[bench]
    fn bench_filter_kernel_low_sigma(b: &mut test::Bencher) {
        b.iter(|| filter_kernel(1.2));
    }

    #[bench]
    fn bench_filter_kernel_high_sigma(b: &mut test::Bencher) {
        b.iter(|| filter_kernel(5.0));
    }

    #[bench]
    fn bench_detect_edges_low_sigma(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        b.iter(|| detect_edges(&image, 1.2));
    }

    #[bench]
    fn bench_detect_edges_high_sigma(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        b.iter(|| detect_edges(&image, 5.0));
    }

    #[bench]
    fn bench_minmax_suppression_low_sigma(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        let edges = detect_edges(&image, 1.2);
        b.iter(|| minmax_suppression(&edges));
    }

    #[bench]
    fn bench_minmax_suppression_high_sigma(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        let edges = detect_edges(&image, 5.0);
        b.iter(|| minmax_suppression(&edges));
    }

    #[bench]
    fn bench_hysteresis(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        let edges = detect_edges(&image, 1.2);
        let edges = minmax_suppression(&edges);
        b.iter(|| hysteresis(&edges, 0.4, 0.1));
    }
}
