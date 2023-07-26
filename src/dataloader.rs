use std::{fs, path};
use std::path::PathBuf;
use rayon::prelude::*;
use tch::vision::image;
use tch::Tensor;
use tch::Kind;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

pub trait DataLoader {
    fn load_batch(&mut self) -> Option<Batch>;
}

pub struct FsDataLoader {
    pub batch_size: i64,
    pub current_batch: i64,
    pub data: Vec<FsDataRef>,
}

#[derive(Debug)]
pub struct Batch {
    pub index: i64,
    pub images: Tensor,
    pub labels: Tensor
}

#[derive(Debug)]
pub struct FsDataRef {
    pub image_path: PathBuf,
    pub label: Tensor
}

// unsafe because chatgtp told me to... to make par iter on dataloading work (seems to be fine?)
unsafe impl Sync for FsDataRef {}

impl FsDataRef {
    pub fn new(image_path: PathBuf, label: Tensor) -> Self {
        FsDataRef {
            image_path,
            label
        }
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
    delimiter: String
}

impl FsLabelDelimiter {
    pub fn new(label: i64, label_name: String, delimiter: String) -> Self {
        Self {
            label,
            label_name,
            delimiter
        }
    }
}


impl DataLoader for FsDataLoader {
 fn load_batch(&mut self) -> Option<Batch> {
        let max_batch_index = (self.data.len() as usize / self.batch_size as usize) as usize;
        let start_idx : usize= (self.current_batch * self.batch_size) as usize;
        let mut end_index : usize= (self.current_batch * self.batch_size + self.batch_size) as usize;
        let data = self.data[start_idx as usize..end_index as usize].to_vec();
        self.current_batch = self.current_batch + 1; // inc batch idx

        let images: Vec<Tensor> = data.par_iter().map(|data| {
            let img = image::load(&data.image_path).unwrap();
            let image = image::resize(&img, 224, 224).unwrap().to_kind(Kind::Float) / 255.0;
            image
        }).collect();


        let labels: Vec<_> = data.iter().map(|data| data.label.copy()).collect();

        let images = Tensor::stack(&images, 0);
        let labels = Tensor::stack(&labels, 0);
        // I think this is correct, need to verify
        if self.current_batch as usize == max_batch_index {
            self.current_batch = 0;
            return None
        }

        let batch = Batch {
            index: self.current_batch,
            images,
            labels
        };

        Some(batch)

        
    }
}

impl FsDataLoader {
    
    pub fn new(directory: &str, batch_size: i64, delimiters: Vec<FsLabelDelimiter>) -> Self {
        let paths: Vec<PathBuf> = fs::read_dir(directory).unwrap()
            .map(|path| path.unwrap().path()).collect();
        
        let mut hash: HashMap<String, Vec<FsDataRef>> = HashMap::new();
    
        for p in paths {
            for d in &delimiters {
                let p_clone = p.clone();
                let path_string = p_clone.to_str().unwrap();
                let contains = path_string.contains(d.delimiter.as_str());

                if contains {
                    let label_tensor = Tensor::from(d.label); //TODO needs to be dyn
                    let data_ref = FsDataRef::new(p.clone(), label_tensor);
                    hash.entry(d.label_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(data_ref);
                }
              
            }
        }
    

        let mut data: Vec<FsDataRef> = vec![];

        for (k,v) in hash.iter() {
      
           data.extend(v.clone()); 
        }
    
        data.shuffle(&mut thread_rng());

        let current_batch = 0;

        Self {
            batch_size,
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