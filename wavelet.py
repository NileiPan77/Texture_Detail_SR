import numpy as np
import matplotlib.pyplot as plt
import pywt
import pyexr
import cv2

def wavelet_png(input_img, output_dir):
    input = cv2.imread(input_img)
    img_gray = np.dot(input[...,:3], [0.2989, 0.5870, 0.1140])

    # Perform 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(img_gray, 'bior2.2')

    # coeffs is a tuple, with coeffs[0] being the approximation coefficients array and coeffs[1] containing the details coefficients in (cH, cV, cD) format.
    cA, (cH, cV, cD) = coeffs

    # threshold
    threshold = np.std(cD) * 0.5

    # Apply soft thresholding
    cD = pywt.threshold(cD, threshold, mode='soft')
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')

    # Perform 2D Inverse Discrete Wavelet Transform
    img_wav = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    plt.imshow(img_wav, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(output_dir + 'cA.png')

def wavelet_exr(input_img, output_dir):
    input = pyexr.open(input_img).get()[:, :, 0]

    wavelet_func = 'dmey'
    # Perform 2D Discrete Wavelet Transform with multiple levels
    coeffs = pywt.wavedec2(input, wavelet_func, level=2)

    # multiplier
    multiplier = 4

    # Apply soft thresholding to each set of detail coefficients
    for i in range(1, len(coeffs)):
        # Each item of coeffs[i] is a details coefficients array in the form (cH, cV, cD)
        cH, cV, cD = coeffs[i]

        # Compute the thresholds
        threshold_cD = np.std(cD) * multiplier
        threshold_cH = np.std(cH) * multiplier
        threshold_cV = np.std(cV) * multiplier

        # Apply the thresholds
        cD = pywt.threshold(cD, threshold_cD, mode='soft')
        cH = pywt.threshold(cH, threshold_cH, mode='soft')
        cV = pywt.threshold(cV, threshold_cV, mode='soft')

        # Store the thresholded coefficients back into the list
        coeffs[i] = (cH, cV, cD)

        multiplier /= 2

    # Perform 2D Inverse Discrete Wavelet Transform
    img_wav = pywt.waverec2(coeffs, wavelet_func)

    # stack the channels
    img_wav = np.stack((img_wav, img_wav, img_wav), axis=-1)
    pyexr.write(output_dir + 'img_wav_' + wavelet_func + '.exr', img_wav)


def edge_detect(img_in, img_out):
    img = pyexr.open(img_in).get()
    # offset 0.5 and normalize
    img = (img - 0.48)
    img = cv2.normalize(img,  None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_edge = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    cv2.imwrite(img_out, img_edge)

if __name__ == '__main__':
    input_img = './data/cavity.exr'
    output_dir = './data/wavelet/'
    # wavelet_png(input_img, output_dir)
    wavelet_exr(input_img, output_dir)
    # edge_detect(input_img, output_dir)