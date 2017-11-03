use std::*;
use std::f32::consts::*;
use image::{self, GenericImage};
use rayon::prelude::*;


const TAU: f32 = PI * 2.0;

#[inline(always)]
fn clamp<T: PartialOrd>(f: T, lo: T, hi: T) -> T {
    debug_assert!(lo < hi);
    if f > hi {
        hi
    } else if f < lo {
        lo
    } else {
        f
    }
}

/// The result of a computation.
#[derive(Clone)]
pub struct Detection {
    edges: Vec<Vec<Edge>>,
}

impl Detection {
    /// Returns the width of the computed image.
    pub fn width(&self) -> usize {
        self.edges.len()
    }

    /// Returns the height of the computed image.
    pub fn height(&self) -> usize {
        self.edges[0].len()
    }

    /// Renders the detected edges to an image.
    ///
    /// The intensity of the pixel represents the magnitude of the change in brightnes while the
    /// color represents the direction.
    ///
    /// Useful for debugging.
    pub fn as_image(&self) -> image::DynamicImage {
        let img = image::RgbImage::from_fn(self.width() as u32, self.height() as u32, |x, y| {
            let (h, s, v) = {
                let edge = &self[(x as usize, y as usize)];
                ((edge.angle() + TAU) % TAU, 1.0, edge.magnitude())
            };
            let (r, g, b) = {
                // http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
                let c = v * s;
                let x = c * (1.0 - ((h / FRAC_PI_3) % 2.0 - 1.0).abs());
                let m = v - c;
                let (r, g, b) = match h {
                    h if h < FRAC_PI_3            => (c, x, 0.0),
                    h if h < FRAC_PI_3 * 2.0      => (x, c, 0.0),
                    h if h < PI                   => (0.0, c, x),
                    h if h < PI + FRAC_PI_3       => (0.0, x, c),
                    h if h < PI + FRAC_PI_3 * 2.0 => (x, 0.0, c),
                    h if h < TAU                  => (c, 0.0, x),
                    _                             => unreachable!(),
                };
                (r + m, g + m, b + m)
            };
            image::Rgb { data: [
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ]}
        });
        image::DynamicImage::ImageRgb8(img)
    }
}

impl ops::Index<usize> for Detection {
    type Output = Edge;
    fn index(&self, index: usize) -> &Self::Output {
        let x = index % self.width();
        let y = index / self.height();
        &self.edges[x][y]
    }
}

impl ops::Index<(usize, usize)> for Detection {
    type Output = Edge;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.edges[index.0][index.1]
    }
}


/// The computed result for a single pixel.
#[derive(Copy, Clone, Debug)]
pub struct Edge {
    vec_x: f32,
    vec_y: f32,
    magnitude: f32,
}

impl Edge {
    fn new(vec_x: f32, vec_y: f32) -> Edge {
        let vec_x = FRAC_1_SQRT_2 * clamp(vec_x, -1.0, 1.0);
        let vec_y = FRAC_1_SQRT_2 * clamp(vec_y, -1.0, 1.0);
        let magnitude = f32::hypot(vec_x, vec_y);
        debug_assert!(0.0 <= magnitude && magnitude <= 1.0);
        let frac_1_mag = if magnitude != 0.0 {
            magnitude.recip()
        } else {
            1.0
        };
        Edge {
            vec_x: vec_x * frac_1_mag,
            vec_y: vec_y * frac_1_mag,
            magnitude,
        }
    }

    /// The direction of the gradient in radians.
    ///
    /// This is a convenience function for `atan2(direction)`.
    pub fn angle(&self) -> f32 {
        f32::atan2(self.vec_y, self.vec_x)
    }

    /// Returns a normalized vector of the direction of the change in brightness
    ///
    /// The vector will point away from the detected line.
    /// E.g. a vertical line separating a dark area on the left and light area on the right will
    /// have it's direction point towards the light area on the right.
    pub fn dir_norm(&self) -> (f32, f32) {
        (self.vec_x, self.vec_y)
    }

    /// The absolute magnitude of the change in brightness.
    ///
    /// Either 0 or 1.
    pub fn magnitude(&self) -> f32 {
        self.magnitude
    }
}


/// Computes the canny edges of an image.
///
/// The variable `sigma` determines the size of the filter kernel which affects the precision and
/// SNR of the computation:
///
/// * A small sigma (3.0<) creates a kernel which is able to discern fine details but is more prone
///   to noise.
/// * Larger values result in detail being lost and are thus best used for detecting large
///   features. Computation time also increases.
///
/// The `weak_threshold` and `strong_threshold` determine what detected pixels are to be regarded
/// as edges and which should be discarded. They are compared with the absolute magnitude of the
/// change in brightness.
///
/// # Panics:
/// * If either `strong_threshold` or `weak_threshold` are outisde the range of 0 to 1 inclusive.
/// * If `strong_threshold` is less than `weak_threshold`.
/// * If `image` contains no pixels (either it's width or height is 0).
pub fn canny<T: Into<image::GrayImage>>(image: T, sigma: f32, strong_threshold: f32, weak_threshold: f32) -> Detection {
    let gs_image = image.into();
    assert!(gs_image.width() > 0);
    assert!(gs_image.height() > 0);
    let edges = detect_edges(&gs_image, sigma);
    let edges = minmax_suppression(&edges, weak_threshold);
    let edges = hysteresis(&edges, strong_threshold, weak_threshold);
    Detection { edges }
}

/// Calculates a 2nd order 2D gaussian derivative with size sigma.
fn filter_kernel(sigma: f32) -> (usize, Vec<(f32, f32)>) {
    let size = (sigma * 10.0).round() as usize;
    let mul_2_sigma_2 = 2.0 * sigma.powi(2);
    let kernel = (0..size).flat_map(|y| {
        (0..size).map(move |x| {
            let (xf, yf) = (x as f32 - size as f32 / 2.0, y as f32 - size as f32 / 2.0);
            let g = (-(xf.powi(2) + yf.powi(2)) / mul_2_sigma_2).exp() / mul_2_sigma_2;
            (xf * g, yf * g)
        })
    }).collect();
    (size, kernel)
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
    let n = ((theta + TAU) % TAU) / TAU;
    let i = (n * 8.0).round() as usize % 8;
    neighbours[i]
}

/// Computes the edges in an image using the Canny Method.
///
/// `sigma` determines the radius of the Gaussian kernel.
fn detect_edges(image: &image::GrayImage, sigma: f32) -> Vec<Vec<Edge>> {
    let (width, height) = (image.width() as i32, image.height() as i32);
    let (ksize, g_kernel) = filter_kernel(sigma);
    let ks = ksize as i32;
    (0..width).into_par_iter().map(|g_ix| {
        let ix = g_ix;
        let kernel = &g_kernel;
        (0..height).into_par_iter().map(move |iy| {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;

            for kyi in 0..ks {
                let ky = kyi - ks / 2;
                for kxi in 0..ks {
                    let kx = kxi - ks / 2;
                    let k = unsafe {
                        let i = (kyi * ks + kxi) as usize;
                        debug_assert!(i < kernel.len());
                        kernel.get_unchecked(i)
                    };

                    let pix = unsafe {
                        // Clamp x and y within the image bounds so no non-existing borders are be
                        // detected based on some background color outside image bounds.
                        let x = clamp(ix + kx, 0, width - 1);
                        let y = clamp(iy + ky, 0, height - 1);
                        image.unsafe_get_pixel(x as u32, y as u32).data[0] as f32
                    };
                    sum_x += pix * k.0;
                    sum_y += pix * k.1;
                }
            }
            Edge::new(sum_x / 255.0, sum_y / 255.0)
        }).collect()
    })
    .collect()
}

/// Narrows the width of detected edges down to a single pixel.
fn minmax_suppression(edges: &Vec<Vec<Edge>>, weak_threshold: f32) -> Vec<Vec<Edge>> {
    let (width, height) = (edges.len(), edges[0].len());
    (0..width).into_par_iter().map(|x| {
        (0..height).into_par_iter().map(|y| {
            let edge = edges[x][y];
            if edge.magnitude < weak_threshold {
                // Skip distance computation for non-edges.
                return Edge::new(0.0, 0.0);
            }

            // A vector confined to a box instead of a radius.
            // The magnitude ranges from 1 to sqrt(2).
            let box_vec = {
                let r = edge.vec_x.abs().recip()
                    .min(edge.vec_y.abs().recip());
                (
                    edge.vec_x * r,
                    edge.vec_y * r,
                )
            };
            debug_assert!((1.0 - box_vec.0.abs()).abs() <= 1e-6 || (1.0 - box_vec.1.abs()).abs() <= 1e-6);
            debug_assert!({
                let m = f32::hypot(box_vec.0, box_vec.1);
                1.0 <= m && m <= SQRT_2
            });

            // Truncating the edge magnitudes helps mitigate rounding errors for thick edges.
            let truncate = |f: f32| (f * 1e5).round() * 1e-6;

            // Find out the current pixel represents the highest, most intense, point of an edge by
            // traveling in a direction perpendicular to the edge to see if there are any more
            // intense edges that are supposedly part of the current edge.
            //
            // We travel in both directions concurrently, this enables us to stop if one side
            // extends longer than the other, greatly improving performance.
            let mut select = 0;
            let mut select_flip_bit = 1;

            // The parameters and variables for each side.
            let inversion = [1.0, -1.0];
            let mut distances = [0i32; 2];
            let mut seek_positions = [(x as f32, y as f32); 2];
            let mut seek_magnitudes = [truncate(edge.magnitude); 2];

            while (distances[0] - distances[1]).abs() <= 1 {
                let seek_pos = &mut seek_positions[select];
                let seek_magnitude = &mut seek_magnitudes[select];

                seek_pos.0 += box_vec.0 * inversion[select];
                seek_pos.1 += box_vec.1 * inversion[select];
                let (nb_a, nb_b, n) = if seek_pos.0.abs().fract() < seek_pos.1.abs().fract() {
                    // X is closest to a point.
                    let x = seek_pos.0.round() as usize;
                    let y1 = clamp(seek_pos.1.floor() as isize, 0, height as isize - 1) as usize;
                    let y2 = clamp(seek_pos.1.ceil() as usize, 0, height - 1);
                    let n = (seek_pos.1.fract() + 1.0).fract();
                    (
                        edges.get(x).map(|col| col[y1]).map(|e| e.magnitude),
                        edges.get(x).map(|col| col[y2]).map(|e| e.magnitude),
                        n,
                    )
                } else {
                    // Y is closest to a point.
                    let y = seek_pos.1.round() as usize;
                    let x1 = clamp(seek_pos.0.floor() as isize, 0, width as isize - 1) as usize;
                    let x2 = clamp(seek_pos.0.ceil() as usize, 0, width - 1);
                    let n = (seek_pos.0.fract() + 1.0).fract();
                    (
                        edges[x1].get(y).map(|e| e.magnitude),
                        edges[x2].get(y).map(|e| e.magnitude),
                        n,
                    )
                };
                debug_assert!(n >= 0.0);

                let interpolated_magnitude = truncate(nb_a.unwrap_or(0.0).mul_add((1.0 - n), nb_b.unwrap_or(0.0) * n));
                let trunc_edge_magnitude = truncate(edge.magnitude);
                // Keep searching until either:
                let end =
                    // The next edge has a lesser magnitude than the reference edge.
                    interpolated_magnitude < trunc_edge_magnitude
                    // The gradient increases, meaning we are going up against an (other) edge.
                    || *seek_magnitude > trunc_edge_magnitude && interpolated_magnitude < *seek_magnitude
                    // We've crossed the image border.
                    || nb_a.is_none() || nb_b.is_none();
                *seek_magnitude = interpolated_magnitude;
                distances[select] += 1;

                // Switch to the other side.
                select ^= select_flip_bit;
                if end {
                    if select_flip_bit == 0 {
                        break;
                    }
                    // After switching to the other side, we set the XOR bit to 0 so we stay there.
                    select_flip_bit = 0;
                }
            }

            // Equal distances denote the middle of the edge.
            // A deviation of 1 is allowed for edges over two equal pixels, in which case, the
            // outer edge (near the dark side) is preferred.
            let is_apex =
                // The distances are equal, the edge's width is odd, making the apex lie on a
                // single pixel.
                distances[0] == distances[1]
                // There is a difference of 1, the edge's width is even, spreading the apex over
                // two pixels. This is a special case to handle edges that run along either the X- or X-axis.
                || (distances[0] - distances[1] == 1 && ((1.0 - edge.vec_x.abs()).abs() < 1e-5 || (1.0 - edge.vec_y.abs()).abs() < 1e-5));
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

/// Links lines together and discards noise.
fn hysteresis(edges: &Vec<Vec<Edge>>, strong_threshold: f32, weak_threshold: f32) -> Vec<Vec<Edge>> {
    assert!(0.0 < strong_threshold && strong_threshold < 1.0);
    assert!(0.0 < weak_threshold && weak_threshold < 1.0);
    assert!(weak_threshold < strong_threshold);

    let (width, height) = (edges.len(), edges.first().unwrap().len());
    let mut edges_out: Vec<Vec<Edge>> = vec![vec![Edge::new(0.0, 0.0); height]; width];
    for x in 0..width {
        for y in 0..height {
            if edges[x][y].magnitude < strong_threshold || edges_out[x][y].magnitude >= strong_threshold {
                continue;
            }

            // Follow along the edge along both sides, preserving all edges which magnitude is at
            // least weak_threshold.
            for side in [0.0, PI].into_iter() {
                let mut current_pos = (x, y);
                loop {
                    let edge = edges[current_pos.0][current_pos.1];
                    edges_out[current_pos.0][current_pos.1] = Edge { magnitude: 1.0, .. edge };
                    // Attempt to find the next line-segment of the edge in tree directions ahead.
                    let (nb_pos, nb_magnitude) = [FRAC_PI_4, 0.0, -FRAC_PI_4].into_iter()
                        .map(|bearing| {
                            neighbour_pos_delta(edge.angle() + FRAC_PI_2 + side + bearing)
                        })
                        // Filter out hypothetical neighbours that are outside image bounds.
                        .filter_map(|(nb_dx, nb_dy)| {
                            let nb_x = current_pos.0 as i32 + nb_dx;
                            let nb_y = current_pos.1 as i32 + nb_dy;
                            if 0 <= nb_x && nb_x < width as i32 && 0 <= nb_y && nb_y < height as i32 {
                                let nb = (nb_x as usize, nb_y as usize);
                                Some((nb, edges[nb.0][nb.1].magnitude))
                            } else {
                                None
                            }
                        })
                        // Select the neighbouring edge with the highest magnitude as the next
                        // line-segment.
                        .fold(((0, 0), 0.0), |(max_pos, max_mag), (pos, mag)| {
                            if mag > max_mag {
                                (pos, mag)
                            } else {
                                (max_pos, max_mag)
                            }
                        });
                    if nb_magnitude < weak_threshold || edges_out[nb_pos.0][nb_pos.1].magnitude >= strong_threshold {
                        break;
                    }
                    current_pos = nb_pos;
                }
            }
        }
    }
    edges_out
}


#[cfg(test)]
mod tests {
    use super::*;

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
        let intermediage_d = Detection { edges };
        let mut fd = fs::File::create(format!("{}.0-vectors.png", path)).unwrap();
        intermediage_d.as_image().save(&mut fd, image::ImageFormat::PNG).unwrap();
        edges_to_image(&intermediage_d.edges).save(format!("{}.1-edges.png", path)).unwrap();
        let edges = minmax_suppression(&intermediage_d.edges, weak_threshold);
        edges_to_image(&edges).save(format!("{}.2-minmax.png", path)).unwrap();
        let edges = hysteresis(&edges, strong_threshold, weak_threshold);
        edges_to_image(&edges).save(format!("{}.3-hysteresis.png", path)).unwrap();
        let detection = Detection { edges };
        let mut fd = fs::File::create(format!("{}.4-result.png", path)).unwrap();
        detection.as_image().save(&mut fd, image::ImageFormat::PNG).unwrap();
        detection
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
            let (ksize, kernel) = filter_kernel(sigma);
            assert!(ksize.pow(2) == kernel.len());
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for (gx, gy) in kernel {
                sum_x += gx;
                sum_y += gy;
            }
            println!("sum = ({}, {}), sigma = {}, kernel_size = {}", sum_x, sum_y, sigma, ksize);
            assert!(-0.0001 < sum_x && sum_x <= 0.0001);
            assert!(-0.0001 < sum_y && sum_y <= 0.0001);
        }
    }

    /// Tests whether a vertical line of 1px wide exists in the middle of the image.
    ///
    /// Returns the location of the line on the X-axis.
    fn detect_vertical_line(detection: &Detection) -> usize {
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
        line_x
    }

    #[test]
    fn detect_vertical_line_simple() {
        let d = canny_output_stages("testdata/line-simple.png", 1.2, 0.4, 0.05);
        let x = detect_vertical_line(&d);
        // The direction of the line's surface normal should follow the X-axis.
        for y in 0..d.height() {
            assert!(d.edges[x][y].angle().abs() < 1e-5);
        }
    }

    #[test]
    fn detect_vertical_line_fuzzy() {
        let d = canny_output_stages("testdata/line-fuzzy.png", 2.0, 0.4, 0.05);
        let x = detect_vertical_line(&d);
        // The direction of the line's surface normal should follow the X-axis.
        for y in 0..d.height() {
            assert!(d.edges[x][y].angle().abs() < 0.01);
        }
    }

    #[test]
    fn detect_vertical_line_weakening() {
        let d = canny_output_stages("testdata/line-weakening.png", 1.2, 0.7, 0.05);
        detect_vertical_line(&d);
        // The line vectors are not tested because they are distorted by the gradient.
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
        b.iter(|| minmax_suppression(&edges, 0.01));
    }

    #[bench]
    fn bench_minmax_suppression_high_sigma(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        let edges = detect_edges(&image, 5.0);
        b.iter(|| minmax_suppression(&edges, 0.01));
    }

    #[bench]
    fn bench_hysteresis(b: &mut test::Bencher) {
        let image = image::open(IMG_PATH).unwrap().to_luma();
        let edges = detect_edges(&image, 1.2);
        let edges = minmax_suppression(&edges, 0.1);
        b.iter(|| hysteresis(&edges, 0.4, 0.1));
    }
}
