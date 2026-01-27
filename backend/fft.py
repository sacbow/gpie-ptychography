from .array import xp

def fft2c(x):
    xp_ = xp()
    return xp_.fft.fftshift(
        xp_.fft.fft2(xp_.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )

def ifft2c(x):
    xp_ = xp()
    return xp_.fft.fftshift(
        xp_.fft.ifft2(xp_.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )
