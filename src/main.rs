extern crate serde;
extern crate bincode;

#[macro_use]
extern crate serde_derive;

extern crate byteorder;
extern crate flate2;
extern crate nalgebra as na;
extern crate reqwest;
extern crate rand;

mod mnist;
mod linear_model;

use mnist::MnistData;
use linear_model::LinearModel;

use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let data = MnistData::open_or_download("./data");
    let mut model = LinearModel::new(28*28, 10);

    println!("Correct percentage: {:.3} %, Eval: {:.3} %", model.eval_correct(&data.training) * 100f32, model.eval_correct(&data.test) * 100f32);

    for epoch in 1..50 {
    	for _ in 1..100 {
    		model.sgd(0.05f32, 500, &data.training);
    	}
    	
    	let corr_per = model.eval_correct(&data.training);
    	println!("[{}] Cost: {}, Correct percentage: {:.3} %", epoch, model.eval_cost(&data.training), corr_per * 100f32);

    	model.save_to_file(&format!("./checkpoints/model_{:.5}_{}.ml", corr_per, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()));
    }

    println!("Correct percentage: {:.3} %, Eval: {:.3} %", model.eval_correct(&data.training) * 100f32, model.eval_correct(&data.test) * 100f32);
}