use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs, path};
use tch::vision::image;
use tch::Kind;
use tch::Tensor;

use crate::augmentation_pipeline::{self, AugmentationPipeline};
use crate::transform_pipeline::TransformPipeline;
use crate::utils::generate_random_numbers;

pub trait DataLoader {
    fn load_batch(&mut self) -> Option<Batch>;
}

pub struct FsDataLoader {
    pub batch_size: i64,
    pub current_batch: i64,
    pub data: Vec<FsDataRef>,
    pub transform_pipeline: TransformPipeline,
    pub augmentation_pipeline: Option<AugmentationPipeline>,
}

#[derive(Debug)]
pub struct Batch {
    pub index: i64,
    pub images: Tensor,
    pub labels: Tensor,
}

#[derive(Debug)]
pub struct FsDataRef {
    pub image_path: PathBuf,
    pub label: Tensor,
}

// make par iter on dataloading work (seems to be fine?)
unsafe impl Sync for FsDataRef {}

impl FsDataRef {
    pub fn new(image_path: PathBuf, label: Tensor) -> Self {
        FsDataRef { image_path, label }
    }
}

impl Clone for FsDataRef {
    fn clone(&self) -> Self {
        Self {
            image_path: self.image_path.clone(),
            label: self.label.shallow_clone(),
        }
    }
}

#[derive(Clone)]
pub struct FsLabelDelimiter {
    label: i64,
    label_name: String,
    delimiter: String,
}

impl FsLabelDelimiter {
    pub fn new(label: i64, label_name: String, delimiter: String) -> Self {
        Self {
            label,
            label_name,
            delimiter,
        }
    }
}

impl DataLoader for FsDataLoader {
    fn load_batch(&mut self) -> Option<Batch> {
        let batch_size: usize;
        let mut start_index: usize;
        let mut end_index: usize;
        let current_batch = self.current_batch as usize;
        let batch_size = self.batch_size as usize;
        let max_batch_index: usize;
        let mut augmentation_data: Vec<FsDataRef> = Vec::new();
        let total_images: usize;
        match &self.augmentation_pipeline {
            Some(pipeline) => {
                let num_augments = pipeline.augmentations.len();
                let images_per_augment = 1 + num_augments as usize; // 1 + the actual number of augments
                total_images = images_per_augment * self.data.len();
                let regular_batch = batch_size / images_per_augment;
                let augmented_batch = batch_size - regular_batch;
                let numbers = generate_random_numbers(
                    augmented_batch,
                    0,
                    self.data.len().try_into().unwrap(),
                    None,
                );
                max_batch_index = (self.data.len() * images_per_augment / batch_size);
                start_index = (current_batch * (batch_size - augmented_batch)) as usize;
                end_index = (current_batch * (batch_size - augmented_batch)
                    + (batch_size - augmented_batch)) as usize;
                augmentation_data = numbers
                    .iter()
                    .map(|&number| self.data[number as usize].clone())
                    .collect();
            }
            None => {
                max_batch_index = self.data.len() / batch_size;
                total_images = self.data.len();
                start_index = current_batch * batch_size;
                end_index = current_batch * batch_size + batch_size;
            }
        }

        //println!("Start idx: {} End idx: {} Max Idx {}", start_index, end_index, total_images);
        if start_index > self.data.len() {
            end_index = self.data.len();
        }

        let data = self.data[start_index..end_index].to_vec();
        self.current_batch = self.current_batch + 1; // inc batch idx

        let mut images: Vec<Tensor> = Vec::new();
        let mut labels: Vec<Tensor> = Vec::new();

        if self.augmentation_pipeline.is_some() {
            let aug_data: Vec<(Tensor, Tensor)> = augmentation_data
                .par_iter()
                .map(|data_ref| {
                    let img = image::load(&data_ref.image_path).unwrap();
                    let augments = self.augmentation_pipeline.clone().unwrap().augment(&img); // can be n augments

                    let mut imgs = Vec::new();
                    augments.iter().for_each(|aug| {
                        let trans = self
                            .transform_pipeline
                            .clone()
                            .transform(aug.shallow_clone());
                        imgs.push((trans, data_ref.label.shallow_clone()));
                    });

                    imgs
                })
                .flatten()
                .collect();

            let data: Vec<(Tensor, Tensor)> = data
                .par_iter()
                .map(|data| {
                    let img = image::load(&data.image_path).unwrap();
                    let trans = self
                        .transform_pipeline
                        .clone()
                        .transform(img.shallow_clone());
                    (trans, data.label.shallow_clone())
                })
                .collect();

            for ((img1, label1), (img2, label2)) in aug_data.iter().zip(data.iter()) {
                images.push(img1.shallow_clone());
                images.push(img2.shallow_clone());
                labels.push(label1.shallow_clone());
                labels.push(label2.shallow_clone());
            }
        } else {
            let data: Vec<(Tensor, Tensor)> = data
                .par_iter()
                .map(|data| {
                    let img = image::load(&data.image_path).unwrap();
                    let trans = self
                        .transform_pipeline
                        .clone()
                        .transform(img.shallow_clone());
                    (trans, data.label.shallow_clone())
                })
                .collect();

            for (img, label) in data.iter() {
                images.push(img.shallow_clone());
                labels.push(label.shallow_clone());
            }
        }

        let images = Tensor::stack(&images, 0);
        let labels = Tensor::stack(&labels, 0);

        if self.current_batch as usize == max_batch_index {
            self.current_batch = 0;
            return None;
        }

        let batch = Batch {
            index: self.current_batch,
            images,
            labels,
        };

        Some(batch)
    }
}

impl FsDataLoader {
    pub fn new(
        directory: &str,
        batch_size: i64,
        delimiters: Vec<FsLabelDelimiter>,
        transform_pipeline: TransformPipeline,
        augmentation_pipeline: Option<AugmentationPipeline>,
    ) -> Self {
        let paths: Vec<PathBuf> = fs::read_dir(directory)
            .unwrap()
            .map(|path| path.unwrap().path())
            .collect();

        let mut hash: HashMap<String, Vec<FsDataRef>> = HashMap::new();

        for p in paths {
            for d in &delimiters {
                let p_clone = p.clone();
                let path_string = p_clone.to_str().unwrap();
                let contains = path_string.contains(d.delimiter.as_str());

                if contains {
                    let label_tensor = Tensor::from(d.label);
                    let data_ref = FsDataRef::new(p.clone(), label_tensor);
                    hash.entry(d.label_name.to_string())
                        .or_insert_with(Vec::new)
                        .push(data_ref);
                }
            }
        }

        let mut data: Vec<FsDataRef> = vec![];

        for (k, v) in hash.iter() {
            data.extend(v.clone());
        }

        data.shuffle(&mut thread_rng());

        let current_batch = 0;

        Self {
            batch_size,
            transform_pipeline,
            augmentation_pipeline,
            current_batch,
            data,
        }
    }
}

impl Iterator for FsDataLoader {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.load_batch()
    }
}
