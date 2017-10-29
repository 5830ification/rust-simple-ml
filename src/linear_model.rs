use mnist::MnistSet;

use bincode::{serialize, deserialize, Infinite};
use na::{DMatrix, DVector};

use rand::{thread_rng, sample};

use std::fs::File;
use std::io;
use std::io::Cursor;

#[derive(Debug, Deserialize, Serialize)]
pub struct LinearModel {
	weights: DMatrix<f32>,
	biases: DVector<f32>
}

pub fn sigmoid(x: f32) -> f32 {
	1f32 / (1f32 + (-x).exp())
}

pub fn sigmoid_prime(x: f32) -> f32 {
	let sx = sigmoid(x);
	sx * (1f32 - sx)
}	
 
impl LinearModel {
	
	/// Create a new linear model, weights and biases are
	/// assigned randomly from [-0.5..0.5]
	pub fn new(input_count: usize, output_count: usize) -> Self {
		let mut weights = DMatrix::new_random(output_count, input_count);
		let mut biases = DVector::new_random(output_count);

		weights.apply(|x| x - 0.5f32);
		biases.apply(|x| x - 0.5f32);

		LinearModel {
			weights,
			biases
		}
	}

	pub fn save_to_file(&self, path: &str) {
		let mut f = File::create(path).unwrap();
		let encoded: Vec<u8> = serialize(self, Infinite).unwrap();

		let mut cursor = Cursor::new(encoded);

		io::copy(&mut cursor, &mut f).unwrap();
	}

	/// Compute the network's prediction for @param input
	pub fn predict(&self, input: &DVector<f32>) -> DVector<f32> {
		let mut res = &self.weights * input + &self.biases;
		res.apply(|x| sigmoid(x));

		res
	}

	pub fn eval_cost(&self, data: &MnistSet) -> f32 {
		let mut acc = 0f32;
		for sample in data.iter() {
			let pred = self.predict(&sample.2);
			let diff = pred - sample.1;
			acc += diff.dot(&diff);
		}

		acc / (2f32 * data.image_count() as f32)
	}

	pub fn eval_correct(&self, data: &MnistSet) -> f32 {
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

	pub fn backprop(&self, data: (&DVector<f32>, &DVector<f32>)) -> (DMatrix<f32>, DVector<f32>) {
		let input = data.1;
		let z = &self.weights * input + &self.biases;
		let a = z.map(|z| sigmoid(z));

		let sprime_z = z.map(|z| sigmoid_prime(z));
		let delta = (a - data.0).component_mul(&sprime_z);


		(&delta*input.transpose(), delta)
	}

	pub fn gradient_descent(&mut self, lr: f32, data: &MnistSet) {
		let mut nw_acc = DMatrix::<f32>::zeros(self.weights.shape().0, self.weights.shape().1);
		let mut nb_acc = DVector::<f32>::zeros(self.biases.shape().0);

		for input in data.iter() {
			let (nw, nb) = self.backprop((input.1, input.2));

			nw_acc += nw;
			nb_acc += nb;
		}

		let wprime = &self.weights - (lr/data.image_count() as f32) * nw_acc;
		let bprime = &self.biases - (lr/data.image_count() as f32) * nb_acc;

		self.weights = wprime;
		self.biases = bprime;
	}

	pub fn sgd(&mut self, lr: f32, batch_size: usize, data: &MnistSet) {
		let mut nw_acc = DMatrix::<f32>::zeros(self.weights.shape().0, self.weights.shape().1);
		let mut nb_acc = DVector::<f32>::zeros(self.biases.shape().0);

		let mut rng = thread_rng();
		for sample in sample(&mut rng, data.iter(), batch_size) {
			let (nw, nb) = self.backprop((sample.1, sample.2));

			nw_acc += nw;
			nb_acc += nb;
		}

		let wprime = &self.weights - (lr/batch_size as f32) * nw_acc;
		let bprime = &self.biases - (lr/batch_size as f32) * nb_acc;

		self.weights = wprime;
		self.biases = bprime;
	}
}
