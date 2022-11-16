<img src="https://github.com/KCL-BMEIS/niftyreg/blob/master/logo/nifty_reg_logo.png?raw=true" alt="NiftyReg logo" title="NiftyReg" height="80">

# NiftyReg

[![License](https://img.shields.io/github/license/KCL-BMEIS/NiftyReg)](https://github.com/KCL-BMEIS/niftyreg/blob/master/LICENSE.txt)
[![GitHub Actions](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/linux.yml/badge.svg?branch=master)](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/linux.yml?query=branch%3Amaster)
[![GitHub Actions](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/macos.yml/badge.svg?branch=master)](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/macos.yml?query=branch%3Amaster)
[![GitHub Actions](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/windows.yml/badge.svg?branch=master)](https://github.com/KCL-BMEIS/niftyreg/actions/workflows/windows.yml?query=branch%3Amaster)



### WHAT DOES THE PACKAGE CONTAIN?

The code contains programs to perform rigid, affine and non-linear registration
of 2D and 3D images stored as Nifti or Analyze (nii or hdr/img).

The rigid and affine registration are performed using an algorithm initially
presented by Ourselin et al.[1]. The symmetric versions of the rigid and
affine registration have been presented in Modat et al.[2].
The non-linear registration is based on the work is based on the work initially
presented by Rueckert et al.[3]. The current implementation has been presented
in Modat et al.[4].

Ourselin et al.[1] presented an algorithm called Aladin, which is based on
a block-matching approach and a Trimmed Least Square (TLS) scheme. Firstly,
the block matching provides a set of corresponding points between a reference
and a warped floating image. Secondly, using this set of corresponding points,
the best rigid or affine transformation is evaluated. This two-step loop is
repeated until convergence to the best transformation.
In our implementation, we used the normalised cross-correlation between the
reference and warped floating blocks to extract the best correspondence. The
block width is constant and has been set to 4 pixels or voxels. A coarse-to-
ﬁne approach is used, where the registration is ﬁrst performed on down-sampled
images (using a Gaussian pyramid) and finally performed on full resolution
images. The symmetric approach optimises concurrently forward and backward
transformations.
reg aladin is the name of the command to perform rigid or affine registration.

The non-rigid algorithm implementation is based on the Free-From Deformation
presented by Rueckert et al.[3]. However, the algorithm has been re-factored
in order to speed-up registration. The deformation of the floating image is
performed using cubic B-splines to generate the deformation ﬁeld. Concretely,
a lattice of equally spaced control points is defined over the reference image
and moving each point allows to locally modify the mapping to the floating
image. In order to assess the quality of the warping between both input images,
an objective function composed from the Normalised Mutual Information (NMI) and
the Bending-Energy (BE) is used. The objective function value is optimised
using the analytical derivative of both, the NMI and the BE within a conjugate
gradient scheme. The symmetric version of the algorithm takes advantage of
stationary velocity field parametrisation.
reg f3d is the command to perform non-linear registration.

A third program, called reg resample, is been embedded in the package. It
uses the output of reg aladin and reg f3d to apply transformation, generate
deformation ﬁelds or Jacobian map images for example.

The code has been implemented for CPU and GPU architecture. The former
code is based on the C/C++ language, whereas the later is based on CUDA
(http://www.nvidia.com).

The nifti library (http://nifti.nimh.nih.gov/) is used to read and write
images. The code is thus dealing with nifti and analyse formats.

If you are planning to use any of our research, we would be grateful if you
would be kind enough to cite reference(s) 2 (rigid or affine) and/or
4 (non-rigid).

### REFERENCES

[1] Ourselin, et al. (2001). Reconstructing a 3D structure from serial
histological sections. Image and Vision Computing, 19(1-2), 25–31.

[2] Modat, et al. (2014). Global image registration using a symmetric block-
matching approach. Journal of Medical Imaging, 1(2), 024003–024003.
doi:10.1117/1.JMI.1.2.024003

[3] Rueckert, et al.. (1999). Nonrigid registration using free-form
deformations: Application to breast MR images. IEEE Transactions on Medical
Imaging, 18(8), 712–721. doi:10.1109/42.796284

[4] Modat, et al. (2010). Fast free-form deformation using graphics processing
units. Computer Methods And Programs In Biomedicine,98(3), 278–284.
doi:10.1016/j.cmpb.2009.09.002


