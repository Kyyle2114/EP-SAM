import torch

def extract_freq_components(x: torch.Tensor) -> torch.Tensor:
    """ Extract High frequency components using FFT """
    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    
   
    W, H = x.shape[-2:]
    center_w, center_h = W // 2, H // 2
    fft[:, :, center_w-1:center_w+1, center_h-1:center_h+1] = 0
    
    high_fr, high_fi = fft.real, fft.imag
    high_fft_hires = torch.fft.ifftshift(torch.complex(high_fr, high_fi))
    high_inv = torch.fft.ifft2(high_fft_hires, norm="forward").real
    high_inv = torch.abs(high_inv)

    return high_inv