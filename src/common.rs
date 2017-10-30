use na::{DMatrix, DVector};

use mnist::MnistSet;

pub fn sigmoid(z: f32) -> f32 {
	1f32 / (1f32 + (-z).exp())
}

pub fn sigmoid_prime(x: f32) -> f32 {
	let sx = sigmoid(x);
	sx * (1f32 - sx)
}

pub trait Model {
	fn predict(&self, input: &DVector<f32>) -> DVector<f32>;

	fn eval_cost(&self, data: &MnistSet) -> f32 {
		let mut acc = 0f32;
		for sample in data.iter() {
			let pred = self.predict(&sample.2);
			let diff = pred - sample.1;
			acc += diff.dot(&diff);
		}

		acc / (2f32 * data.image_count() as f32)
	}

	fn eval_correct(&self, data: &MnistSet) -> f32 {
		let mut correct_count = 0;
		for sample in data.iter() {
			let pred = self.predict(&sample.2);
			let label = sample.0;

			let pred_val = pred.iter().enumerate().max_by(|&(_, x), &(_, b)| x.partial_cmp(b).unwrap()).unwrap().0;
			if pred_val == label as usize {
				correct_count += 1;
			}
		}

		correct_count as f32 / data.image_count() as f32
	}
}


