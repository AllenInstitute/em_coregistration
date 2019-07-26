from alignment.transform import Transform
from alignment.data_filter import DataFilter
from alignment.data_handler import DataLoader
import json
import numpy as np

#boss_template_url1 = "https://neuroglancer.theboss.io/#!{'layers':{'transformed':{'type':'image'_'source':'boss://https://api.theboss.io/minnie/fine_aligned/em'}}_'navigation':{'pose':{'position':{'voxelSize':[8_8_40]_'voxelCoordinates':[XXX_YYY_ZZZ]}}_'zoomFactor':ZOOM}_'perspectiveZoom':50_'showSlices':false_'layout':'xy'}"

def nglink1(template_url, xyz, zoomFactor=100): 
    link = str(template_url) 
    link = link.replace('XXX', '%d' % xyz[0]) 
    link = link.replace('YYY', '%d' % xyz[1]) 
    link = link.replace('ZZZ', '%d' % xyz[2]) 
    link = link.replace('ZOOM', '%0.1f' % zoomFactor) 
    return link 

example = {
  'optical' : {
          'dset1': {
              'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
              'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
              'actions': ['invert_opty'],
              'sd_set': {'src': 'opt', 'dst': 'em'}
              },
          'dset_soma': {
              'landmark_file': './data/landmarks_somata_final.csv',
              'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
              'actions': ['invert_opty'],
              'sd_set': {'src': 'opt', 'dst': 'em'}
              },
          'dset2': {
              'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize.csv',
              'header': ['optz', 'opty', 'optx'],
              'actions': ['opt_px_to_mm'],
              'sd_set': {'src': 'opt', 'dst': 'em'}
              },
          'output_file': './tmp_out/filtered_tmp.csv',
          'header': 'opt',
          },
  'filter_with' : {
          'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize.csv',
          'header': ['optz', 'opty', 'optx'],
          'sd_set': {'src': 'opt', 'dst': 'em'}
          },
  'transform_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/tmp_out/transform.json',
  'template_url': 'https://neuromancer-seung-import.appspot.com/#!{"layers":[{"tab":"annotations","selectedAnnotation":"data-bounds","source":"precomputed://gs://microns-seunglab/minnie_v4/alignment/fine/sergiy_multimodel_v1/vector_fixer30_faster_v01/image_stitch_multi_block_v1","type":"image","name":"Minnie65"}],"navigation":{"pose":{"position":{"voxelSize":[4,4,40],"voxelCoordinates":[XXX, YYY, ZZZ]}},"zoomFactor":ZOOM},"jsonStateServer":"https://www.dynamicannotationframework.com/nglstate/post","layout":"4panel"}'
  }

#orig = DataLoader(input_data=de, args=[])
#orig.run()
#
#tfj = '/allen/programs/celltypes/workgroups/em-connectomics/danielk/3D_alignment/delivery_2019_04_15/transform.json'
#
#baylor_mm = DataFilter(input_data=filtere, args=[])
#baylor_mm.run()
#
#tf = Transform(model='TPS')
#with open(tfj, 'r') as f:
#    tf.from_dict(json.load(f))
#
def flip_y_mm(vec):
    vec[1] = 1.322 - vec[1]
    return vec

#dst = tf.transform(baylor_mm.closest)
#closest_output = []
#in_index = np.argwhere(baylor_mm.inside).flatten()
#for ir in np.arange(dst.shape[0]):
#    closest_output.append({})
#    closest_output[-1]['em_link'] = nglink1(dst[ir])
#    closest_output[-1]['em_nm (x, y, z)'] = np.round(dst[ir], 2).tolist()
#    closest_output[-1]['opt_mm (x, y, z)'] = np.round(baylor_mm.newdata[ir], 4).tolist()
#    closest_output[-1]['opt_mm fiji (x, 1.322 - y, z)'] = np.round(flip_y_mm(baylor_mm.newdata[ir]), 4).tolist()
#    closest_output[-1]['original_px (z, y, x)'] = orig.data['src'][in_index[ir]][::-1].tolist()
#
#dst = tf.transform(baylor_mm.newdata)
#all_output = []
#in_index = np.argwhere(baylor_mm.inside).flatten()
#for ir in np.arange(dst.shape[0]):
#    all_output.append({})
#    all_output[-1]['em_link'] = nglink1(dst[ir])
#    all_output[-1]['em_nm (x, y, z)'] = np.round(dst[ir], 2).tolist()
#    all_output[-1]['opt_mm (x, y, z)'] = np.round(baylor_mm.newdata[ir], 4).tolist()
#    all_output[-1]['opt_mm fiji (x, 1.322 - y, z)'] = np.round(flip_y_mm(baylor_mm.newdata[ir]), 4).tolist()
#    all_output[-1]['original_px (z, y, x)'] = orig.data['src'][in_index[ir]][::-1].tolist()

#with open('closest.csv', 'w') as f:
#    for o in closest_output:
#        f.write('%0.2f, %0.2f, %0.2f, %0.3f, %0.3f, %0.3f\n' % (
#            o['em_nm (x, y, z)'][0],
#            o['em_nm (x, y, z)'][1],
#            o['em_nm (x, y, z)'][2],
#            o['opt_mm (x, y, z)'][0],
#            o['opt_mm (x, y, z)'][1],
#            o['opt_mm (x, y, z)'][2]))
#
#with open('all.csv', 'w') as f:
#    for o in all_output:
#        f.write('%0.2f, %0.2f, %0.2f, %0.3f, %0.3f, %0.3f\n' % (
#            o['em_nm (x, y, z)'][0],
#            o['em_nm (x, y, z)'][1],
#            o['em_nm (x, y, z)'][2],
#            o['opt_mm (x, y, z)'][0],
#            o['opt_mm (x, y, z)'][1],
#            o['opt_mm (x, y, z)'][2]))


class MakeLinks():
    def __init__(self, example):
        self.orig = DataLoader(
                input_data=example['filter_with'], args=[])
        self.orig.run()
        
        self.optical = DataFilter(input_data=example['optical'], args=[])
        self.optical.run()
        
        self.tf = Transform(model='TPS')
        with open(example['transform_json'], 'r') as f:
            self.tf.from_dict(json.load(f))

        self.template = example['template_url']

    def run(self):
        #dst = tf.transform(baylor_mm.closest)
        in_index = np.argwhere(self.optical.inside).flatten()
        self.closest_output = self.create_link(self.optical.closest, in_index)
        self.all_output = self.create_link(self.optical.newdata, in_index)

        #for ir in np.arange(dst.shape[0]):
        #    closest_output.append({})
        #    closest_output[-1]['em_link'] = nglink1(dst[ir])
        #    closest_output[-1]['em_nm (x, y, z)'] = np.round(dst[ir], 2).tolist()
        #    closest_output[-1]['opt_mm (x, y, z)'] = np.round(baylor_mm.newdata[ir], 4).tolist()
        #    closest_output[-1]['opt_mm fiji (x, 1.322 - y, z)'] = np.round(flip_y_mm(baylor_mm.newdata[ir]), 4).tolist()
        #    closest_output[-1]['original_px (z, y, x)'] = orig.data['src'][in_index[ir]][::-1].tolist()
        #
        #dst = tf.transform(baylor_mm.newdata)
        #all_output = []
        #in_index = np.argwhere(baylor_mm.inside).flatten()
        #for ir in np.arange(dst.shape[0]):
        #    all_output.append({})
        #    all_output[-1]['em_link'] = nglink1(dst[ir])
        #    all_output[-1]['em_nm (x, y, z)'] = np.round(dst[ir], 2).tolist()
        #    all_output[-1]['opt_mm (x, y, z)'] = np.round(baylor_mm.newdata[ir], 4).tolist()
        #    all_output[-1]['opt_mm fiji (x, 1.322 - y, z)'] = np.round(flip_y_mm(baylor_mm.newdata[ir]), 4).tolist()
        #    all_output[-1]['original_px (z, y, x)'] = orig.data['src'][in_index[ir]][::-1].tolist()

    def create_link(self, data, in_index):
        dst = self.tf.transform(data)
        output = []
        for ir in np.arange(dst.shape[0]):
            output.append({})
            output[-1]['em_link'] = nglink1(self.template, dst[ir])
            output[-1]['em_nm (x, y, z)'] = np.round(dst[ir], 2).tolist()
            output[-1]['opt_mm (x, y, z)'] = np.round(self.optical.newdata[ir], 4).tolist()
            output[-1]['opt_mm fiji (x, 1.322 - y, z)'] = np.round(flip_y_mm(self.optical.newdata[ir]), 4).tolist()
            output[-1]['original_px (z, y, x)'] = self.orig.data['src'][in_index[ir]][::-1].tolist()
        return output


if __name__ == "__main__":
    m = MakeLinks(example)
    m.run()
