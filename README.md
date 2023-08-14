# Texture Detail SR

The main idea of this algorithm is to transfer fine details like pores and wrinkles from a high resolution texture to a noisy scanned texture.

To achieve this, I used Fourier transform and converted image to frequency domain for easier transfer process. Then I used Butterworth filter for both high pass and low pass filtering, as it is often referred to as a **maximally flat magnitude filter** and thus provide a smooth transition around the cutoff frequency while behave ripple-less.

Finally, apply high pass filter to the high-res texture to extract the small details (rapid changes), low pass filter to the low-res texture to extract the shape and large features of input texture (gradual changes).

<img src=".\1024px-Filters_order5.svg.png" alt="1024px-Filters_order5.svg" style="zoom:50%;" />

## Usage

```bash
python feature_transfer.py [section_name]
```

where `[section_name]` refers to the sections inside `config.ini` which have format:

```ini
[DEFAULT]
base_diffuse = .\target\base_diffive.png
base_cavity = .\data\cavity.exr
target_diffuse = .\target\target_diffuse.png
target_height = .\data\height_16k.exr
output_dir = ./target/results/
align = False
cutoff_high = 120
cutoff_low = 180
degree = 1
output_name = out_cavity.exr
```

`[DEFAULT]`: Section name

`base_diffuse`: Path to the diffuse map of the ***low-res*** texture, `str`

`base_cavity`: Path to the cavity map of the ***low-res*** texture, `str`

`target_diffuse`: Path to the diffuse map of the ***high-res*** texture, `str`

`target_height`: Path to the height map of the ***high-res*** texture, `str`

`output_dir`: Output path, `str`

`align`: Set to True if target maps and base map are ***NOT*** aligned, `Boolean`

`cutoff_high`: Cutoff frequency for high pass filter, `int`

`cutoff_low`: Cutoff frequency for low pass filter, `int`

`degree`: Degree of the Butterworth filter, higher the degree, more rapid changes around the cutoff frequency, `int`

`output_name`: Output name of the result cavity map, `str`





## 