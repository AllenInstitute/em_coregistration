em_coregistration
#################

tools for:

- handling data from Baylor and Fiji Image stacks
- performing a 3D alignment solve between landmarks
- transforming points based on a 3D solve result
- create neuroglancer links for checking visualization
 
support
#######

We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support, as it is under active development. The community is welcome to submit issues, but you should not expect an active response.

Acknowledgement of Government Sponsorship
#########################################

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.


User Guide
##########

from the root dir of this repo (or anywhere if installed):
::
   python fit_and_predict.py

   average residual [dst units]: 10803.2841
   average residual [dst units]: 5626.8712
   average residual [dst units]: 4696.0970
   average residual [dst units]: 3975.3966
   average residual [dst units]: 3480.0259
   average residual [dst units]: 3053.2877
   average residual [dst units]: 2877.6895
   average residual [dst units]: 3.0340
   transform 0 control points moved average of 1407.9um
   transform 1 control points moved average of 18.0um
   transform 2 control points moved average of 6.6um
   transform 3 control points moved average of 12.5um
   transform 4 control points moved average of 7.5um
   transform 5 control points moved average of 3.8um
   transform 6 control points moved average of 1.6um
   transform 7 control points moved average of 2.9um
   worst points
      Pt-1729      134.9
      Pt-3159      134.3
      Pt-2155      124.4
      Pt-1610      124.1
      Pt-3094      116.6
       Pt-415      116.5
      Pt-2138      109.9
      Pt-2136      109.0
      Pt-3782       87.2
      Pt-1024       86.8
   wrote data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Master_updated.csv

this just performed a staged solve, showing residuals and control point motions for the specified transform steps. Refer to fit_and_predict.py for more details.

Running this can be time-consuming:
::
    python leave_one_out.py

For testing, one can change the leave-out fraction inside the file to something smaller than 1 (for example 0.002 will jsut run a few). I tend to run it on a cluster node. See coreg.pbs.

.. The neuroglancer voxels are anisotropic, but the Fiji coordinates are isotropic. It is easier to just solve and transform in isotropic coordinates. From the transform results, it is an additional step to go to voxels:
   ::
      from coregister.transform import em_nm_to_voxels
   
      em_nm_to_voxels(s2.data['dst'])[0:4]
   
      array([[290095, 176880,  14977],
             [344856, 145878,  16363],
             [342623, 187225,  17086],
             [318735, 124452,  15965]])
   
   you can go backwards also:
   ::
      em_nm_to_voxels(em_nm_to_voxels(s2.data['dst']), inverse=True)[0:4]
   
      array([[1172668.,  717760.,  282120.],
             [1391712.,  593752.,  337560.],
             [1382780.,  759140.,  366480.],
             [1287228.,  508048.,  321640.]])
   
   There is a not-so-smooth way to make a neuroglancer link:
   ::
      from links.make_ndviz_links import nglink1, example
      vox = em_nm_to_voxels(s2.data['dst'])[0:4]
      vox
   
      array([[290095, 176880,  14977],
             [344856, 145878,  16363],
             [342623, 187225,  17086],
             [318735, 124452,  15965]])
   
      print(nglink1(example['template_url'], vox[0]))
   
      https://neuromancer-seung-import.appspot.com/#!{"layers":[{"tab":"annotations","selectedAnnotation":"data-bounds","source":"precomputed://gs://microns-seunglab/minnie_v4/alignment/fine/sergiy_multimodel_v1/vector_fixer30_faster_v01/image_stitch_multi_block_v1","type":"image","name":"Minnie65"}],"navigation":{"pose":{"position":{"voxelSize":[4,4,40],"voxelCoordinates":[290095, 176880, 14977]}},"zoomFactor":100.0},"jsonStateServer":"https://www.dynamicannotationframework.com/nglstate/post","layout":"4panel"}
