use na::{DMatrix, DVector};

use common::{sigmoid, sigmoid_prime, Model};
use mnist::MnistSet;

#[derive(Debug, Deserialize, Serialize)]
pub struct NeuralNetwork {
	pub layers: Vec<Layer>
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Layer {
	weights: DMatrix<f32>,
	biases: DVector<f32>
}

pub struct LayerIter<'a> {
	idx: usize,
	net: &'a NeuralNetwork
}

impl<'a> Iterator for LayerIter<'a> {
	type Item = &'a Layer;

	fn next(&mut self) -> Option<Self::Item> {
		if self.idx >= self.net.layer_count() {
			None
		} else {
			let l = &self.net.layers[self.idx];
			self.idx += 1;

			Some(l)
		}
	}
}

impl NeuralNetwork {
	pub fn with_layer_sizes(layer_sizes: &[usize])  {
		if layer_sizes.len() < 2 {
			panic!("Layer sizes need to be at least [input, output]");
		}

		let mut layers: Vec<Layer> = Vec::new();
		for (idx, size) in layer_sizes.iter().enumerate().skip(1) {				// Kinda make this more pretty some day...
			let mut weights = DMatrix::new_random(*size, layer_sizes[idx - 1]);
			let mut biases = DVector::new_random(*size);

			weights.apply(|x| x - 0.5f32);
			biases.apply(|x| x - 0.5f32);

			layers.push(Layer {
				weights,
				biases
			});
		}
	}

	pub fn layer_iter(&self) -> LayerIter {
		LayerIter {
			idx: 0,
			net: &self
		}
	}

	pub fn layer_count(&self) -> usize {
		self.layers.len()
	}
}

impl Model for NeuralNetwork {
	fn predict(&self, input: &DVector<f32>) -> DVector<f32> {
		let mut prev_a = input.clone();

		for layer in self.layer_iter() {
			prev_a = &layer.weights * prev_a + &layer.biases;
			prev_a.apply(|a| sigmoid(a));
		}

		prev_a
	}

	fn gradient_descent(&mut self, lr: f32, data: &MnistSet) {
		unimplemented!();
	}

	fn sgd(&mut self, lr: f32, batch_size: usize, data: &MnistSet) {
		unimplemented!();
	}
}