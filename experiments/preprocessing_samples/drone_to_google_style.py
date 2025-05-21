import cv2 as cv
import numpy as np
from pathlib import Path

def load_image(path):
    img = cv.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def lab_histogram_transfer(source, target):
    source_lab = cv.cvtColor(source, cv.COLOR_BGR2LAB)
    target_lab = cv.cvtColor(target, cv.COLOR_BGR2LAB)
    s_l, s_a, s_b = cv.split(source_lab)
    t_l, t_a, t_b = cv.split(target_lab)
    s_hist = cv.calcHist([s_l], [0], None, [256], [0,256]).cumsum()
    t_hist = cv.calcHist([t_l], [0], None, [256], [0,256]).cumsum()
    s_hist = s_hist * 255 / s_hist[-1]
    t_hist = t_hist * 255 / t_hist[-1]
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 0
        while j < 256 and s_hist[i] > t_hist[j]:
            j += 1
        lut[i] = j
    l = cv.LUT(s_l, lut)
    result_lab = cv.merge([l, s_a, s_b])
    return cv.cvtColor(result_lab, cv.COLOR_LAB2BGR)

def apply_colour_matrix(img, M):
    img_f = img.astype(np.float32)
    img_f = img_f @ M.T
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)

def detect_shadow(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    shadow_mask = (l < 128).astype(np.float32)
    shadow_mask = cv.GaussianBlur(shadow_mask, (15, 15), 0)
    return shadow_mask

def gaussian_soften(image, sigma=1.2):
    return cv.GaussianBlur(image, (0, 0), sigma)

def downsample_to_match(img, target_shape):
    return cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_AREA)

def main():
    base = Path(__file__).parent
    drone_path = base / 'drone.png'
    google_path = base / 'google.png'
    out_path = base / 'processed' / 'drone_google_style.jpg'
    out_path.parent.mkdir(exist_ok=True)

    img = load_image(drone_path)
    google = load_image(google_path)

    # (1) Undistort + homography (skip if K, D, H not available)
    try:
        from camera_params import K, D, H, W_out, H_out
        img = cv.undistort(img, K, D)
        img = cv.warpPerspective(img, H, (W_out, H_out))
    except ImportError:
        print('Camera parameters not found, skipping undistort/homography.')
    except Exception as e:
        print(f'Error in undistort/homography: {e}. Skipping.')

    # (2) Colour style transfer
    img = lab_histogram_transfer(img, google)

    # (3) Optional: apply colour matrix (identity if not provided)
    try:
        from camera_params import M_3x3
        img = apply_colour_matrix(img, M_3x3)
    except ImportError:
        pass
    except Exception as e:
        print(f'No colour matrix applied: {e}')

    # (4) Shadow softening
    S = detect_shadow(img)
    img = img / (1 - 0.5 * S)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # (5) Blur & resample
    img = gaussian_soften(img, sigma=1.2)
    img = downsample_to_match(img, google.shape)

    cv.imwrite(str(out_path), img, [int(cv.IMWRITE_JPEG_QUALITY), 85])
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main() 