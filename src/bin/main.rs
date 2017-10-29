extern crate clap;
extern crate ml;

use ml::mnist::MnistData;
use ml::linear_model::LinearModel;

use clap::{Arg, App};

use std::fs::metadata;
use std::fs::DirBuilder;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let matches = App::new("Rust ML")
        .arg(Arg::with_name("checkpoint file")
            .short("l")
            .long("load")
            .help("Load a previous checkpoint")
            .takes_value(true))
        .arg(Arg::with_name("learning rate")
            .short("n")
            .long("learning-rate")
            .help("Set the gradient descent learning rate (defaults to 0.05)")
            .takes_value(true))
        .get_matches();

    let data = MnistData::open_or_download("./data");
    let mut model: LinearModel;
    let lr = matches.value_of("learning rate").map(|s| s.parse::<f32>().unwrap()).unwrap_or(0.05f32);

    create_checkpoint_dir();

    if let Some(checkpoint_file) = matches.value_of("checkpoint file") {
        model = LinearModel::from_file(checkpoint_file);
    } else {
        model = LinearModel::new(28*28, 10);
    }

    println!("Correct percentage: {:.3} %, Eval: {:.3} %", model.eval_correct(&data.training) * 100f32, model.eval_correct(&data.test) * 100f32);

    for epoch in 1..25 {
    	for _ in 1..200 {
    		model.sgd(lr, 500, &data.training);
    	}
    	
    	let corr_per = model.eval_correct(&data.training);
    	println!("[{}] Cost: {}, Correct percentage: {:.3} %", epoch, model.eval_cost(&data.training), corr_per * 100f32);

    	model.save_to_file(&format!("./checkpoints/model_{:.5}_{}.ml", corr_per, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()));
    }

    println!("Correct percentage: {:.3} %, Eval: {:.3} %", model.eval_correct(&data.training) * 100f32, model.eval_correct(&data.test) * 100f32);
}

fn create_checkpoint_dir() {
    if metadata("./checkpoints").is_err() {
        DirBuilder::new()
            .create("./checkpoints")
            .unwrap();
    }
}