use image::RgbImage;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use tch::Tensor;

pub fn tensor_to_rgb_image(t: &Tensor) -> RgbImage {
    let image: RgbImage = TryFrom::try_from(t).unwrap();

    image
}

pub fn rgb_image_to_tensor(i: RgbImage) -> Tensor {
    let tensor: Tensor = TryFrom::try_from(&i).unwrap();
    tensor
}

pub fn generate_random_numbers(n: usize, min: i32, max: i32, exclude: Option<&[i32]>) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    let mut numbers = Vec::with_capacity(n);

    while numbers.len() < n {
        let random_number = rng.gen_range(min..max);
        match exclude {
            Some(exclude) => {
                if !exclude.contains(&random_number) {
                    numbers.push(random_number);
                }
            }
            None => {
                numbers.push(random_number);
            }
        }
    }

    numbers
}
