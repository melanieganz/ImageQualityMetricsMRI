import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import normalize_percentile


subject = "sub-02"
acq = "acq-mpragepmcon_rec-wre_run-02"

file = glob.glob(f"OpenNeuro_dataset/{subject}/mprage/**{acq}**")[0]

img = nib.load(file).get_fdata().astype(np.uint16)

img = normalize_percentile(img)

# use plt.imshow to plot one slice of the image and adjust the figure size to the image size
# plt.figure(figsize=(img.shape[1], img.shape[2]))
plt.figure(figsize=(img.shape[0]/10, img.shape[1]/10))
plt.imshow(img[:, ::-1, 150].T, cmap='gray', vmax=0.9)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


input_csv = ("Results/OpenNeuro/2024-11-20_13-26/ImageQualityMetricsScores.csv")
data = np.loadtxt(input_csv, delimiter=',',
                  unpack=True, dtype=str)
subjects_header = data[0]
header = data[:, 0]
ind_subject = np.where(subjects_header == subject)[0]
data_subj = data[:, ind_subject]
acq_ind = np.where(data_subj ==[s for s in data_subj[1] if acq in s][0])
data_acq = data_subj[:, acq_ind[1]]

print("Subject: ", subject, " Acquisition: ", acq, "Slice: 150")
print("Observer scores: ")
for i in range(12, 16):
    print(data[i, 0], ": ", data_acq[i][0])
print("Average:", 1/6*(np.float32(data_acq[12, 0][0])
                       + np.float32(data_acq[13, 0][0])
                       + 2*np.float32(data_acq[14, 0][0])
                       + 2*np.float32(data_acq[15, 0][0])))

print("Image quality metrics: ")
for i in range(2, 12):
    print(data[i, 0], ": ", data_acq[i][0])


print("Done")





