// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Image decoding to interleaved RGB8.
//!
//! With the default `libjpeg` feature, JPEGs go through libjpeg (mozjpeg, a
//! libjpeg-turbo fork: ISLOW IDCT and fancy upsampling, as PIL decodes
//! them), so the engine sees the pixels the official preprocessing sees. The
//! `image` crate's JPEG decoder (zune-jpeg) differs from PIL by a few levels
//! in a sizeable share of samples; it is the fallback without the feature
//! and the decoder for PNG and WebP. [`crate::preprocess::decode`] adds
//! PIL's alpha compositing on top.

use std::path::Path;

use anyhow::{Context, Result};

/// Interleaved 8-bit RGB, row-major from the top-left.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgb8Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; width as usize * height as usize * 3],
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// Reads and decodes an image file ([`decode_bytes`]).
pub fn decode_file(path: &Path) -> Result<Rgb8Image> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode_bytes(&bytes).with_context(|| format!("decode {}", path.display()))
}

/// Decodes a JPEG, PNG or WebP to RGB8; alpha and 16-bit samples are
/// dropped or reduced, greyscale is expanded.
pub fn decode_bytes(bytes: &[u8]) -> Result<Rgb8Image> {
    #[cfg(feature = "libjpeg")]
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return decode_jpeg_libjpeg(bytes);
    }
    decode_bytes_image_crate(bytes)
}

/// The `image` crate's decoders (zune-jpeg for JPEG), for formats libjpeg
/// does not read and for comparisons.
pub fn decode_bytes_image_crate(bytes: &[u8]) -> Result<Rgb8Image> {
    let img = image::load_from_memory(bytes)?.into_rgb8();
    Ok(Rgb8Image {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    })
}

/// libjpeg decode with PIL's defaults. libjpeg reports errors by unwinding,
/// so the call runs under `catch_unwind`.
#[cfg(feature = "libjpeg")]
pub fn decode_jpeg_libjpeg(bytes: &[u8]) -> Result<Rgb8Image> {
    let decoded = std::panic::catch_unwind(|| -> std::io::Result<Rgb8Image> {
        let mut d = mozjpeg::Decompress::new_mem(bytes)?;
        d.dct_method(mozjpeg::DctMethod::IntegerSlow);
        d.do_fancy_upsampling(true);
        let mut started = d.rgb()?;
        let (width, height) = (started.width() as u32, started.height() as u32);
        let data = started.read_scanlines::<u8>()?;
        started.finish()?;
        Ok(Rgb8Image {
            width,
            height,
            data,
        })
    });
    match decoded {
        Ok(Ok(img)) => {
            anyhow::ensure!(
                img.data.len() == img.width as usize * img.height as usize * 3,
                "libjpeg returned {} bytes for {}x{}",
                img.data.len(),
                img.width,
                img.height
            );
            Ok(img)
        }
        Ok(Err(e)) => Err(e).context("libjpeg"),
        Err(_) => Err(anyhow::anyhow!("libjpeg failed to decode the JPEG")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(img: &Rgb8Image, format: ::image::ImageFormat) -> Vec<u8> {
        let mut out = std::io::Cursor::new(Vec::new());
        ::image::RgbImage::from_raw(img.width, img.height, img.data.clone())
            .unwrap()
            .write_to(&mut out, format)
            .unwrap();
        out.into_inner()
    }

    fn gradient(w: u32, h: u32) -> Rgb8Image {
        let mut img = Rgb8Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                img.data[i] = (x * 255 / (w - 1)) as u8;
                img.data[i + 1] = (y * 255 / (h - 1)) as u8;
                img.data[i + 2] = 128;
            }
        }
        img
    }

    #[test]
    fn png_round_trips_exactly() {
        let img = gradient(37, 21);
        let got = decode_bytes(&encode(&img, ::image::ImageFormat::Png)).unwrap();
        assert_eq!(got, img);
    }

    #[test]
    fn jpeg_decoders_agree_closely() {
        let img = gradient(64, 48);
        let jpeg = encode(&img, ::image::ImageFormat::Jpeg);
        let a = decode_bytes(&jpeg).unwrap();
        let b = decode_bytes_image_crate(&jpeg).unwrap();
        assert_eq!((a.width, a.height), (64, 48));
        let max = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap();
        assert!(max <= 8, "libjpeg vs zune-jpeg max diff {max}");
        assert_eq!(a.pixel(0, 0).len(), 3);
    }
}
