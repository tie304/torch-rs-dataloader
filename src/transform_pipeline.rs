use std::sync::Arc;
use tch::vision::image as tch_img;
use tch::{Kind, TchError, Tensor};

use image::{DynamicImage, GenericImageView, GrayImage, RgbImage};

type Transform = Arc<dyn Fn(Tensor) -> Tensor + Send + Sync>;

#[derive(Clone)]
pub struct TransformPipeline {
    transforms: Vec<Transform>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }
    pub fn add_transform(&mut self, transform: Transform) {
        self.transforms.push(transform);
    }
    pub fn transform(self, img: Tensor) -> Tensor {
        self.transforms
            .iter()
            .fold(img, |output, transform| (transform.as_ref())(output))
    }
}

pub fn resize(img: Tensor, w: i64, h: i64) -> Tensor {
    tch_img::resize(&img, w, h).expect("Could not resize image")
}

pub fn to_float(img: Tensor) -> Tensor {
    img.to_kind(Kind::Float)
}

pub fn normalize(img: Tensor) -> Tensor {
    img / 255.0
}

pub fn to_cuda(img: Tensor) -> Tensor {
    img.to(tch::Device::Cuda(0))
}

//TODO super slow.. Why?
pub fn greyscale(t: Tensor) -> Tensor {
    // Convert the Tensor into a RgbImage.
    let image: RgbImage = TryFrom::try_from(&t).unwrap();

    // Convert the RgbImage to grayscale.
    let grayscale_image: GrayImage = image::imageops::grayscale(&image);

    // Convert the grayscale image back into a Tensor.
    let grayscale_tensor: Tensor = TryFrom::try_from(&grayscale_image).unwrap();

    grayscale_tensor
}
