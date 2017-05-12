#####################
# NIFTY_REG PACKAGE #
#####################

##############################################################################

--------------------------------
1 WHAT DOES THE PACKAGE CONTAIN?
--------------------------------
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

##############################################################################

---------
2 LICENSE
---------
Copyright (c) 2009, University College London, United-Kingdom
All rights reserved.

Redistribution and use in floating and binary forms, with or without
modification,
are permitted provided that the following conditions are met:

Redistributions of floating code must retain the above copyright notice,
this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of the University College London nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

##############################################################################

---------
3 CONTACT
---------
For any comment, please, feel free to contact Marc Modat (m.modat@ucl.ac.uk).

##############################################################################

------------
4 REFERENCES
------------
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

##############################################################################
##############################################################################
##############################################################################
##############################################################################

