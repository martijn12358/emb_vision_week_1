import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

img = plt.imread("picture_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

hist = ndi.histogram(gray, 0,255, 256)
hist_n = hist/hist.sum()
fig, axes = plt.subplots(2, 2)
axes[1][0].plot(hist, label='Histogram')
axes[1][1].plot(hist_n, label='Histogram')
axes[0][0].imshow(img)
axes[0][1].imshow(gray, cmap='gray')
axes[0][0].axis('off')
axes[0][1].axis('off')
plt.show()

img_equ = cv2.equalizeHist(gray)
hist_equ = ndi.histogram(img_equ, 0,255,256)
hist_equ_n = hist_equ/hist_equ.sum()
cdf = hist_n.cumsum()
cdf_n = cdf * float(hist_n.max()) / cdf.max()
cdf_equ = hist_equ_n.cumsum()
cdf_n_equ = cdf_equ * float(hist_equ_n.max()) / cdf_equ.max()

fig, axes = plt.subplots(2, 2)
axes[1][0].plot(hist_n, label='Histogram')
axes[1][0].plot(cdf_n, label='Histogram')
axes[1][1].plot(hist_equ_n, label='Histogram')
axes[1][1].plot(cdf_n_equ, label='Histogram')
axes[0][0].imshow(gray, cmap='gray')
axes[0][1].imshow(img_equ, cmap='gray')
axes[0][0].axis('off')
axes[0][1].axis('off')
plt.show()
