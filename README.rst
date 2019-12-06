.. image:: https://travis-ci.com/AllenInstitute/em_coregistration.svg?branch=master
    :target: https://travis-ci.com/AllenInstitute/em_coregistration
.. image:: https://codecov.io/gh/codecov/em_coregistration/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/codecov/em_coregistration


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

Important Files
###############

* **data/staged_transform_args.json** : input arguments for solve 2P->EM
* **data/inverse_staged_transform_args.json** : the other way
* **data/staged_transform_solution.json** : transform solution 2P -> EM
* **data/inverse_staged_transform_solution.json** : the other way
* **data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Masterleave_outs.json** : leave-one-out residuals
* **data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Masterinverse_leave_outs.json** : the other way

User Guide
##########

solve and use transform
-----------------------
::

    import json
    import coregister.solve as cs
    with open("./data/staged_transform_args.json", "r") as f: 
        j =json.load(f)                      
    s = cs.Solve3D(input_data=j, args=['--output_json', 'solved_transform_out.json'])                                       
    s.run()                                                                                                                 

    from coregister.transform.transform import Transform          
    with open('./solved_transform_out.json', 'r') as f: 
        tfj = json.load(f)
    t = Transform(json=tfj)                                                                                                 
    src = s.data['src'][0:5]                                                                                              
    dst = s.data['dst'][0:5]                                                                                                 
    src                                                                                                                     
    Out[20]: 
    array([[0.804354, 0.387835, 0.274648],
           [0.869191, 0.138524, 0.156319],
           [0.85027 , 0.257353, 0.067352],
           [0.924324, 0.750244, 0.225444],
           [0.934366, 0.182291, 0.188485]])
    dst
    Out[21]:
    array([[1172669.4,  717762.7,  282148.8],
           [1391713. ,  593754.5,  337574.1],
           [1287230.6,  508049.5,  321659.2],
           [ 869550.9,  684943.2,  399641.7],
           [1363565.2,  611139.6,  398796.7]])

    t.tform(src)                                                                                                             
    Out[22]: 
    array([[1172669.3087167 ,  717762.76429371,  282148.60118503],
           [1391715.11237805,  593753.07429628,  337574.56376036],
           [1287230.51081691,  508049.13923649,  321659.30456869],
           [ 869551.3786766 ,  684942.05607145,  399640.98195057],
           [1363565.80927674,  611138.89843287,  398797.29234907]])

more detail for the solving process
-----------------------------------
from the root dir of this repo. If running as an installed package, you'll need to copy this data directory somewhere with r/w permissions.
::
   python fit_and_predict.py

    average residual [dst units]: 3.1620
    worst points
       Pt-3044   0.020422
        Pt-727   0.035287
       Pt-1047   0.042640
       Pt-2182   0.047620
        Pt-236   0.047632
       Pt-2211   0.049158
       Pt-2757   0.053443
       Pt-1862   0.057763
        Pt-768   0.058496
      Pt-25103   0.058756
    PolynomialModel residual 9932.802718
    ChunkedModel residual 7296.263690
    ChunkedModel residual 5909.392451
    SplineModel residual 4503.411981 27 cntrls moved 22585.214699
    SplineModel residual 3781.220908 125 cntrls moved 18947.336478
    SplineModel residual 3173.745507 1000 cntrls moved 6918.062524
    SplineModel residual 2893.618486 1728 cntrls moved 3360.592100
    SplineModel residual 3.161994 2936 cntrls moved 2892.010575
    wrote data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Master_updated.csv


this just performed a staged solve, showing residuals and control point motions for the specified transform steps. Refer to fit_and_predict.py for more details.

Running this can be time-consuming:
::
    python leave_one_out.py

For testing, one can change the leave-out fraction inside the file to something smaller than 1 (for example 0.002 will run just a few). I tend to run it on a cluster node. See coreg.pbs.

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
