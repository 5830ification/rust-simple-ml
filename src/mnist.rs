use std::fs;
use std::fs::File;
use std::fs::DirBuilder;

use std::io;
use std::io::Cursor;

use byteorder::{ReadBytesExt, BigEndian};

use flate2::read::GzDecoder;

use na::DVector;

use reqwest;

const DATA_URL_BASE: &str = "http://yann.lecun.com/exdb/mnist";
const DATA_FILENAMES: &'static [&'static str] = &["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
			"t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"];

#[derive(Debug)]
pub struct MnistData {
	pub training: MnistSet,
	pub test: MnistSet
}

#[derive(Debug)]
pub struct MnistSet {
	labels_raw: Vec<u8>,
	images_raw: Vec<u8>,
	labels_oh: Vec<DVector<f32>>,
	img_vecs: Vec<DVector<f32>>
}

pub struct MnistSetIter<'a> {
	idx: usize,
	set: &'a MnistSet
}

impl<'a> Iterator for MnistSetIter<'a> {
	type Item = (u8, &'a DVector<f32>, &'a DVector<f32>);

	fn next(&mut self) -> Option<(u8, &'a DVector<f32>, &'a DVector<f32>)> {
		if self.idx >= self.set.image_count() {
			None
		} else {
			let res = (self.set.get_label(self.idx), self.set.get_label_oh(self.idx), self.set.get_img_vec(self.idx));

			self.idx += 1;

			Some(res)
		}
	}
}

impl MnistData {
	pub fn open_or_download(path: &str) -> Self {
		for &file in DATA_FILENAMES.iter() {
			if !fs::metadata(format!("{}/{}", path, file)).is_ok() {
				println!("{}/{} not found! Downloading training data...", path, file);
				Self::download_training_data(path);
				break;
			}
		}

		Self::load_data(path)
	}

	fn download_training_data(destination: &str) {
		DirBuilder::new()
			.create(destination).unwrap();

		for &file in DATA_FILENAMES.iter() {
			let full_url = format!("{}/{}", DATA_URL_BASE, file);
			let full_dest = format!("{}/{}", destination, file);
			Self::download_file(&full_url, &full_dest);
		}
	}

	fn download_file(url: &str, destination: &str) {
		println!("Downloading {}...", url);

		let mut resp = reqwest::get(url).unwrap();
		let mut f = File::create(destination).unwrap();

		io::copy(&mut resp, &mut f).unwrap();
	}

	fn load_data(path: &str) -> Self {
		MnistData {
			training: MnistSet::from_files(&format!("{}/{}", path, DATA_FILENAMES[0]), &format!("{}/{}", path, DATA_FILENAMES[1])),
			test: MnistSet::from_files(&format!("{}/{}", path, DATA_FILENAMES[2]), &format!("{}/{}", path, DATA_FILENAMES[3]))
		}
	}
}

impl MnistSet {
	pub fn from_files(imgfile: &str, labelfile: &str) -> Self {
		let img_len = fs::metadata(imgfile).unwrap().len();
		let lbl_len = fs::metadata(labelfile).unwrap().len();
		let mut images_raw = Vec::with_capacity(img_len as usize);
		let mut labels_raw = Vec::with_capacity(lbl_len as usize);

		let im_file = File::open(imgfile).unwrap();
		let lbl_file = File::open(labelfile).unwrap();

		let mut gz = GzDecoder::new(im_file).unwrap();
		io::copy(&mut gz, &mut images_raw).unwrap();

		let mut gz = GzDecoder::new(lbl_file).unwrap();
		io::copy(&mut gz, &mut labels_raw).unwrap();

		let mut set = MnistSet {
			images_raw,
			labels_raw,
			labels_oh: vec![],
			img_vecs: vec![]
		};

		if set.magic_numbers() != (2049, 2051) {
			println!("{}, {}", set.magic_numbers().0, set.magic_numbers().1);
			panic!("Malformed input data: invalid magic numbers!");
		}

		set.labels_oh = Vec::with_capacity(set.image_count());
		set.img_vecs = Vec::with_capacity(set.image_count());

		for idx in 0..set.image_count() {
			let lbl = set.get_label(idx);
			let mut vec = DVector::zeros(10);

			vec[lbl as usize] = 1f32;
			set.labels_oh.push(vec);

			let img_vec = DVector::from_iterator(
				28*28, 
				set.get_img(idx)
					.iter()
					.map(|v| *v as f32 / 255f32)
			);
			set.img_vecs.push(img_vec);
		}

		set
	}

	pub fn magic_numbers(&self) -> (u32, u32) {
		let lbl_magic = (&*self.labels_raw).read_u32::<BigEndian>().unwrap();
		let img_magic = (&*self.images_raw).read_u32::<BigEndian>().unwrap(); 
		
		(lbl_magic, img_magic)
	}

	pub fn image_count(&self) -> usize {
		let mut c = Cursor::new(&self.images_raw);
		c.set_position(4);
		c.read_u32::<BigEndian>().unwrap() as usize
	}

	pub fn get_label(&self, idx: usize) -> u8 {
		self.labels_raw[8 + idx]
	}

	pub fn get_label_oh(&self, idx: usize) -> &DVector<f32> {
		&self.labels_oh[idx]
	}

	pub fn get_img(&self, idx: usize) -> &[u8] {
		&self.images_raw[16 + idx * 28 * 28..(16 + (idx + 1) * 28 * 28)]
	}

	pub fn get_img_vec(&self, idx: usize) -> &DVector<f32> {
		&self.img_vecs[idx]
	}

	pub fn iter(&self) -> MnistSetIter {
		MnistSetIter {
			idx: 0,
			set: self
		}
	}

	pub fn get_pair(&self, idx: usize) -> (&DVector<f32>, &DVector<f32>) {
		(
			self.get_label_oh(idx),
			self.get_img_vec(idx)
		)
	}
}