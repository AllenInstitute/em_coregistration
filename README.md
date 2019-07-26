# em_coregistration

April 2019

For the mm^3 EM data set and the functional data from Baylor

(...maybe CT later ?)

tools for:
- handling data from Baylor and Fiji Image stacks
- performing a 3D alignment solve between landmarks
- transforming points based on a 3D solve result
- create neuroglancer links for checking visualization

# User Guide
using Ipython from the root dir of this repo:

```
> import alignment.solve_3d as s3
> s1 = s3.Solve3D(input_data=s3.example1, args=[])
> s1.run()
average residual [dst units]: 0.0023
```
this just solved for example1 in alignment/solve_3d. The source (moving) = EM, the destination (fixed) is optical. The average residual is 2.3um.

Some of the data for this example:
```
> s1.data['src'][0:4]                                                                                                         array([[1172669.371 ,  717762.7498,  282148.7913],
       [1391713.009 ,  593754.4669,  337574.0701],
       [1382783.763 ,  759141.8597,  366487.6834],
       [1287230.575 ,  508049.4798,  321659.1986]])

> s1.data['dst'][0:4]                                                                                                         array([[0.80435367, 0.38783457, 0.27464807],
       [0.86919081, 0.13852442, 0.15631878],
       [0.89353762, 0.18544056, 0.36127396],
       [0.85027004, 0.25735289, 0.06735198]])
```

The result of the transform:
```
> s1.transform.transform(s1.data['src'])[0:4]                                                                                 array([[0.80430773, 0.38744583, 0.2744871 ],
       [0.86856248, 0.14021783, 0.15375644],
       [0.89270374, 0.18248724, 0.358625  ],
       [0.85024923, 0.25737231, 0.06724511]])
```

One can get the residuals by comparing the last 2 outputs.

The transform has also just been written to disk:
```
> s1.args['output_json']                                                                                                      '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/tmp_out/transform.json'
```

You can use this later without re-doing the solve (though the solve is very fast):
```
> from alignment.transform import Transform
> import json
> t = Transform(model='TPS')
> with open(s1.args['output_json'], 'r') as f:
      t.from_dict(json.load(f))
> t.transform(s1.data['src'])[0:4]
array([[0.80430773, 0.38744583, 0.2744871 ],
       [0.86856248, 0.14021783, 0.15375644],
       [0.89270374, 0.18248724, 0.358625  ],
       [0.85024923, 0.25737231, 0.06724511]])
```

Going the other way (optical to EM):
```
> s2 = s3.Solve3D(input_data=s3.example2, args=[])
> s2.run()
average residual [dst units]: 1608.6795
```
The residuals are better in this direction... not exactly sure why. It could be where the control points get set up. Same deal, you can read this transform from disk:
```
> t = Transform(model='TPS')
> with open(s2.args['output_json'], 'r') as f: 
    t.from_dict(json.load(f))
> t.transform(s2.data['src'])[0:4]
array([[1173037.06239277,  717944.16524402,  281592.01954812],
       [1392944.86317897,  590682.87649827,  338896.28718442],
       [1378151.60056365,  761835.79200597,  367174.83412847],
       [1287215.11926234,  507937.83934919,  321698.63528779]])
> s2.data['dst'][0:4]
array([[1172669.371 ,  717762.7498,  282148.7913],
       [1391713.009 ,  593754.4669,  337574.0701],
       [1382783.763 ,  759141.8597,  366487.6834],
       [1287230.575 ,  508049.4798,  321659.1986]])
```
Looks pretty good...

The neuroglancer voxels are anisotropic, but the Fiji coordinates are isotropic. It is easier to just solve and transform in isotropic coordinates. From the transform results, it is an additional step to go to voxels:

```
> from alignment.transform import em_nm_to_voxels
> em_nm_to_voxels(s2.data['dst'])[0:4]
array([[290095, 176880,  14977],
       [344856, 145878,  16363],
       [342623, 187225,  17086],
       [318735, 124452,  15965]])
```

you can go backwards also:
```
> em_nm_to_voxels(em_nm_to_voxels(s2.data['dst']), inverse=True)[0:4]
array([[1172668.,  717760.,  282120.],
       [1391712.,  593752.,  337560.],
       [1382780.,  759140.,  366480.],
       [1287228.,  508048.,  321640.]])
```

There is a not-so-smooth way to make a neuroglancer link:
```
> from links.make_ndviz_links import nglink1, example
> vox = em_nm_to_voxels(s2.data['dst'])[0:4]
> vox
array([[290095, 176880,  14977],
       [344856, 145878,  16363],
       [342623, 187225,  17086],
       [318735, 124452,  15965]])
> print(nglink1(example['template_url'], vox[0]))
https://neuromancer-seung-import.appspot.com/#!{"layers":[{"tab":"annotations","selectedAnnotation":"data-bounds","source":"precomputed://gs://microns-seunglab/minnie_v4/alignment/fine/sergiy_multimodel_v1/vector_fixer30_faster_v01/image_stitch_multi_block_v1","type":"image","name":"Minnie65"}],"navigation":{"pose":{"position":{"voxelSize":[4,4,40],"voxelCoordinates":[290095, 176880, 14977]}},"zoomFactor":100.0},"jsonStateServer":"https://www.dynamicannotationframework.com/nglstate/post","layout":"4panel"}
```
# support

We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support, as it is under active development. The community is welcome to submit issues, but you should not expect an active response.

# Acknowledgement of Government Sponsorship

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.



