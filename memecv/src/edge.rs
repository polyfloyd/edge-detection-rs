use std::*;
use std::f32::consts::*;
use image;


#[derive(Clone)]
pub struct Detection {
    edges: Vec<Vec<Edge>>,
}

impl Detection {
    fn width(&self) -> usize {
        self.edges.len()
    }

    fn height(&self) -> usize {
        self.edges.first().unwrap().len()
    }
}

#[derive(Copy, Clone)]
pub struct Edge {
    vec_x: f32,
    vec_y: f32,
    magnitude: f32,
    theta: f32,
}

impl Edge {
    fn new(vec_x: f32, vec_y: f32) -> Edge {
        let vec_x = if vec_x > 1.0 {
            1.0
        } else if vec_x < -1.0 {
            -1.0
        } else {
            vec_x
        };
        let vec_y = if vec_y > 1.0 {
            1.0
        } else if vec_y < -1.0 {
            -1.0
        } else {
            vec_y
        };
        Edge {
            vec_x,
            vec_y,
            magnitude: (vec_x.powi(2) + vec_y.powi(2)).sqrt(),
            theta: f32::atan2(vec_x, vec_y),
        }
    }
}

pub fn canny<T: Into<image::GrayImage>>(image: T, sigma: f32, strong_threshold: f32, weak_threshold: f32) -> Detection {
    let edges = detect_edges(image, sigma);
    let edges = minmax_suppression(&edges);
    let edges = hysteresis(&edges, strong_threshold, weak_threshold);
    Detection { edges }
}


fn gaussian_derivative_2d(x: f32, y: f32, sigma: f32) -> (f32, f32) {
    let gx = x * E.powf(-(x.powi(2) + y.powi(2)) / (2. * sigma.powi(2))) / (2.0 * sigma.powi(2));
    let gy = y * E.powf(-(x.powi(2) + y.powi(2)) / (2. * sigma.powi(2))) / (2.0 * sigma.powi(2));
    (gx, gy)
}

/// Calculates the width and height of the kernel produced by `gaussian_derivative_2d`.
fn kernel_size(sigma: f32) -> i32 {
    (sigma * 10.0).round() as i32
}

fn neighbour_pos_delta(theta: f32) -> (i32, i32) {
    let neighbours = [
        (0, -1),  // center top
        (1, -1),  // top right
        (1, 0),   // middle right
        (1, 1),   // bottom right
        (0, 1),   // center bottom
        (-1, 1),  // bottom left
        (-1, 0),  // middle left
        (-1, -1), // top left
    ];
    let ang = ((theta + 2.0 * PI) % (2.0 * PI) / (2.0 * PI)) * 8.0;
    assert!(ang >= 0.0 && ang <= 8.0);
    neighbours[ang.floor() as usize]
}

/// Computes the edges in an image using the Canny Method.
///
/// `sigma` determines the radius of the Gaussian kernel.
fn detect_edges<T: Into<image::GrayImage>>(image: T, sigma: f32) -> Vec<Vec<Edge>> {
    let image = image.into();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut edges: Vec<Vec<Edge>> = vec![vec![Edge::new(0.0, 0.0); height]; width];
    let kernel_size = kernel_size(sigma);
    let edge_borders = 0.5;
    for ix in 0..width as i32 {
        for iy in 0..height as i32 {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for kx in -kernel_size / 2..kernel_size / 2 {
                for ky in -kernel_size / 2..kernel_size / 2 {
                    // Clamp x and y within the image bounds so no non-existing borders are be
                    // detected based on some background color outside image bounds.
                    let x = cmp::min(cmp::max(ix + kx, 0), width as i32 - 1);
                    let y = cmp::min(cmp::max(iy + ky, 0), height as i32 - 1);
                    let pix = image.get_pixel(x as u32, y as u32).data[0] as f32 / 255.0;
                    // TODO: cache kernel
                    let (gx, gy) = gaussian_derivative_2d(kx as f32, ky as f32, sigma);
                    sum_x += pix * gx;
                    sum_y += pix * gy;
                }
            }
            edges[ix as usize][iy as usize] = Edge::new(sum_x, sum_y);
        }
    }
    edges
}

/// Narrows the width of detected edges down to a single pixel
fn minmax_suppression(edges: &Vec<Vec<Edge>>) -> Vec<Vec<Edge>> {
    let (width, height) = (edges.len(), edges.first().unwrap().len());
    let mut edges_out: Vec<Vec<Edge>> = vec![vec![Edge::new(0.0, 0.0); height]; width];
    for x in 0..width {
        for y in 0..height {
            let edge = edges[x][y];
            let magnitude_exceeded = [0.0f32, 1.0].into_iter().fold(false, |magnitude_exceeded, invert| {
                if magnitude_exceeded {
                    return magnitude_exceeded;
                }
                // Determine the neighbours we should interpolate from.
                let nb_a = neighbour_pos_delta(edge.theta + PI * invert + PI);
                let nb_b = neighbour_pos_delta(edge.theta + PI * invert);

                // Now get the magnitudes of the neighbours.
                let nb_a_x = x as i32 + nb_a.0;
                let nb_a_y = y as i32 + nb_a.1;
                let nb_a_magnitude = if nb_a_y < 0 || nb_a_y >= height as i32 || nb_a_x < 0 || nb_a_x >= width as i32 {
                    0.0
                } else {
                    edges[nb_a_x as usize][nb_a_y as usize].magnitude
                };
                let nb_b_x = x as i32 + nb_b.0;
                let nb_b_y = y as i32 + nb_b.1;
                let nb_b_magnitude = if nb_b_y < 0 || nb_b_y >= height as i32 || nb_b_x < 0 || nb_b_x >= width as i32 {
                    0.0
                } else {
                    edges[nb_b_x as usize][nb_b_y as usize].magnitude
                };

                // Interpolate between the two neighbours.
                let theta_q = edge.theta + FRAC_PI_4; // Theta aligned with the corners.
                let n = if theta_q < FRAC_PI_2 { // Right
                    (1.0 - ((edge.vec_y / edge.magnitude + 1.0) % 1.0)) * SQRT_2
                } else if theta_q < PI { // Bottom
                    (1.0 - ((edge.vec_x / edge.magnitude + 1.0) % 1.0)) * SQRT_2
                } else if theta_q < PI + FRAC_PI_2 { // Left.
                    ((edge.vec_y / edge.magnitude + 1.0) % 1.0) * SQRT_2
                } else { // Top
                    ((edge.vec_x / edge.magnitude + 1.0) % 1.0) * SQRT_2
                };
                let nb_mag = nb_a_magnitude * n + nb_b_magnitude * (1.0 - n);
                magnitude_exceeded || edge.magnitude < nb_mag
                // TODO: handle equal magnitudes for neighbours
            });
            if !magnitude_exceeded {
                edges_out[x][y] = edge
            }
        }
    }
    edges_out
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
                    edges_out[top.0][top.1].magnitude = 1.0;

                    for invert in [0.0, PI].into_iter() {
                        let (nb_pos, nb_magnitude) = [-FRAC_PI_4, 0.0, FRAC_PI_4].into_iter()
                            .map(|bearing| {
                                // Add PI/2 to theta so it aligns with the edge.
                                neighbour_pos_delta(edge.theta + FRAC_PI_2 + invert + bearing)
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

fn edges_to_image(edges: &Vec<Vec<Edge>>) -> image::GrayImage {
    let (width, height) = (edges.len(), edges.first().unwrap().len());
    let mut image = image::GrayImage::from_pixel(width as u32, height as u32, image::Luma{ data: [0] });
    for x in 0..width {
        for y in 0..height {
            let edge = edges[x][y];
            *image.get_pixel_mut(x as u32, y as u32) = image::Luma{ data: [(edge.magnitude * FRAC_1_SQRT_2 * 255.0).round() as u8]};
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canny_output_stages<T: AsRef<str>>(path: T) -> Detection {
        let path = path.as_ref();
        let image = image::open(path).unwrap();
        let edges = detect_edges(image.to_luma(), 1.0);
        edges_to_image(&edges).save(format!("{}.0-edges.png", path)).unwrap();
        let edges = minmax_suppression(&edges);
        edges_to_image(&edges).save(format!("{}.1-minmax.png", path)).unwrap();
        let edges = hysteresis(&edges, 0.4, 0.05);
        edges_to_image(&edges).save(format!("{}.2-hysteresis.png", path)).unwrap();
        Detection { edges }
    }

    #[test]
    fn kernel_integral_in_bounds() {
        // The integral for the filter kernel should approximate 0.
        for sigma_i in 1..200 {
            let sigma = sigma_i as f32 / 10.0;
            let ksize = kernel_size(sigma);
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for x in -ksize / 2..ksize / 2 {
                for y in -ksize / 2..ksize / 2 {
                    let (gx, gy) = gaussian_derivative_2d(x as f32, y as f32, sigma);
                    sum_x += gx;
                    sum_y += gy;
                }
            }
            println!("sum = ({}, {}), sigma = {}, kernel_size = {}", sum_x, sum_y, sigma, ksize);
            assert!(-0.05 < sum_x && sum_x <= 0.05);
            assert!(-0.05 < sum_y && sum_y <= 0.05);
        }
    }

    #[test]
    fn detect_vertical_line_simple() {
        // A vertical line of 1px wide should exist in the middle of the image.
        let detection = canny_output_stages("testdata/line-simple.png");
        // Find the line.
        let mut line_x = None;
        for x in 0..detection.width() {
            if detection.edges[x][detection.height() / 2].magnitude > 0.8 {
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
            assert!(detection.edges[line_x][y].magnitude == 1.0);
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
}
