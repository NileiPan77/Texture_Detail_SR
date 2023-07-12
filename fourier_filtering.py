import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
def butterworth_lowpass(cutoff, shape, order=1):
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols) * cols
    y = np.linspace(-0.5, 0.5, rows) * rows
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    mask = 1 / (1.0 + (d / cutoff) ** (2 * order))
    return mask

def butterworth_highpass(cutoff, shape, order=1):
    return 1 - butterworth_lowpass(cutoff, shape, order)

def torch_butterworth_lowpass(cutoff, shape, order=1):
    rows, cols = shape
    x = torch.linspace(-0.5, 0.5, cols) * cols
    y = torch.linspace(-0.5, 0.5, rows) * rows
    x, y = torch.meshgrid(x, y)
    d = torch.sqrt(x**2 + y**2)
    mask = 1 / (1.0 + (d / cutoff) ** (2 * order))
    return mask.cuda()

def torch_butterworth_highpass(cutoff, shape, order=1):
    return 1 - torch_butterworth_lowpass(cutoff, shape, order)

# r: radius of the circle
def rectangle_mask(shape, r):
    rows, cols = shape
    crow, ccol = rows//2 , cols//2
    mask1 = np.ones((rows, cols), np.uint8)
    center=[crow,ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask1[mask_area] = 0

    return mask1

# circular mask for low-pass filtering
def circular_mask(shape, r):
    rows, cols = shape
    crow, ccol = rows//2 , cols//2
    center=[crow,ccol]
    mask2 = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask2[mask_area] = 1

    return mask2

def cufourier_filtering(target_exr, base_exr, r = 180, r_high = 120, degree = 1):
    # Copy the images to the GPU
    target_exr_torch = torch.from_numpy(target_exr).cuda()
    base_exr_torch = torch.from_numpy(base_exr).cuda()

    # Compute the Fourier Transform
    f1 = torch.fft.fftshift(torch.fft.fft2(target_exr_torch))
    f2 = torch.fft.fftshift(torch.fft.fft2(base_exr_torch))
    img_shape = target_exr.shape

    # delete the variables to free the memory
    del target_exr_torch
    del base_exr_torch
    torch.cuda.empty_cache()

    # normalize f1
    # f1 = (f1 - 0.5) * 10 + 0.5
    
    
    highpass_filter = torch_butterworth_highpass(r_high, img_shape, degree)
    lowpass_filter = torch_butterworth_lowpass(r, img_shape, degree)

    # Apply the high-pass filter to image1
    f1_highpass = f1 * highpass_filter * 3

    # Apply the low-pass filter to image2
    f2_lowpass = f2 * lowpass_filter

    # Combine the two frequency-domain images
    combined = f1_highpass + f2_lowpass

    # Shift the zero-frequency component back to the corners and perform the inverse Fourier Transform
    image_combined_torch = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(combined)))
    
    # Convert the result back to a numpy array on the CPU
    image_combined = image_combined_torch.cpu().numpy()

    return image_combined

def cufourier_filtering_3out(target_exr, base_exr, r = 180, r_high = 120, degree = 1, degree_high = 1):
    # Copy the images to the GPU
    target_exr_torch = torch.from_numpy(target_exr).cuda()
    base_exr_torch = torch.from_numpy(base_exr).cuda()

    # Compute the Fourier Transform
    f1 = torch.fft.fftshift(torch.fft.fft2(target_exr_torch))
    f2 = torch.fft.fftshift(torch.fft.fft2(base_exr_torch))
    img_shape = target_exr.shape

    # delete the variables to free the memory
    del target_exr_torch
    del base_exr_torch
    torch.cuda.empty_cache()

    # normalize f1
    # f1 = (f1 - 0.5) * 10 + 0.5
    
    highpass_filter = torch_butterworth_highpass(r_high, img_shape, degree_high)
    lowpass_filter = torch_butterworth_lowpass(r, img_shape, degree)

    # Apply the high-pass filter to image1
    f1_highpass = f1 * highpass_filter * 3

    # Apply the low-pass filter to image2
    f2_lowpass = f2 * lowpass_filter

    # Combine the two frequency-domain images
    combined = f1_highpass + f2_lowpass

    # Shift the zero-frequency component back to the corners and perform the inverse Fourier Transform
    image_combined_torch = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(combined)))
    
    # highpass image
    f1_highpass = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(f1_highpass))).cpu().numpy()

    # lowpass image
    f2_lowpass = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(f2_lowpass))).cpu().numpy()

    # Convert the result back to a numpy array on the CPU
    image_combined = image_combined_torch.cpu().numpy()

    return f1_highpass, f2_lowpass, image_combined

def update_highpass(target_exr, r_high, degree):
    # Copy the images to the GPU
    target_exr_torch = torch.from_numpy(target_exr).cuda()

    # Compute the Fourier Transform
    f1 = torch.fft.fftshift(torch.fft.fft2(target_exr_torch))
    img_shape = target_exr.shape

    # delete the variables to free the memory
    del target_exr_torch
    torch.cuda.empty_cache()
    
    highpass_filter = torch_butterworth_highpass(r_high, img_shape, degree)

    # Apply the high-pass filter to image1
    f1_highpass = f1 * highpass_filter

    # Shift the zero-frequency component back to the corners and perform the inverse Fourier Transform
    f1_highpass = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(f1_highpass))).cpu().numpy()

    return f1_highpass

def update_lowpass(base_exr, r, degree):
    # Copy the images to the GPU
    base_exr_torch = torch.from_numpy(base_exr).cuda()

    # Compute the Fourier Transform
    f2 = torch.fft.fftshift(torch.fft.fft2(base_exr_torch))
    img_shape = base_exr.shape

    # delete the variables to free the memory
    del base_exr_torch
    torch.cuda.empty_cache()

    lowpass_filter = torch_butterworth_lowpass(r, img_shape, degree)

    # Apply the low-pass filter to image2
    f2_lowpass = f2 * lowpass_filter

    # Shift the zero-frequency component back to the corners and perform the inverse Fourier Transform
    f2_lowpass = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(f2_lowpass))).cpu().numpy()

    return f2_lowpass

def fourier_filtering(target_exr, base_exr, r = 180, r_high = 120, degree = 1):
    f1 = np.fft.fftshift(np.fft.fft2(target_exr))
    f2 = np.fft.fftshift(np.fft.fft2(base_exr))
    img_shape = target_exr.shape

    # normalize f1
    f1 = (f1 - 0.5) * 10 + 0.5

    highpass_filter = butterworth_highpass(r_high, img_shape, degree)
    # print("highpass_filter shape: ", highpass_filter.shape)
    lowpass_filter = butterworth_lowpass(r, img_shape, degree)

    
    # Apply the high-pass filter to image1
    f1_highpass = f1 * highpass_filter

    # Apply the high-pass filter to image2
    # f2_highpass = f2 * butterworth_highpass(200, target_exr.shape[:2], degree)[:, :, None]

    # Apply the low-pass filter to image2
    f2_lowpass = f2 * lowpass_filter

    # Combine the two frequency-domain images
    combined = f1_highpass + f2_lowpass

    # Shift the zero-frequency component back to the corners and perform the inverse Fourier Transform
    image_combined = np.abs(np.fft.ifft2(np.fft.ifftshift(combined)))
 

    return image_combined
