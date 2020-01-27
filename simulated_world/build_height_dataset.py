from depth_height import *
from math import tan, radians
from PIL import Image
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simfolder",
    type=str,
    help="Path to the simulated data",
)
parser.add_argument("--output_path", type=str, default=".", help="outputs path")
parser.add_argument("--far", type=int, default=1000, help="Simulator camera far parameter")
opts =  parser.parse_args()


depth_path = os.path.join(opts.simfolder, 'Depth/')
seg_path = os.path.join(opts.simfolder, 'Segmentation/')
im_path = os.path.join(opts.simfolder, 'Normal/')
json_path = os.path.join(opts.simfolder, 'JSON/')

depth_paths = glob.glob(depth_path + '*')

depth_metric_folder = os.path.join(opts.output_path, 'Depth_metric/')
heights_folder = os.path.join(opts.output_path, 'Heights/')
cam_params_folder = os.path.join(opts.output_path, 'Cam_params/')
far = opts.far

if not os.path.exists(depth_metric_folder):
    os.makedirs(depth_metric_folder)
if not os.path.exists(heights_folder):
    os.makedirs(heights_folder)
if not os.path.exists(cam_params_folder):
    os.makedirs(cam_params_folder)
    
for elem in depth_paths:
    im_id = os.path.basename(elem).strip('.png')
    json_file = os.path.join(json_path, im_id + '.json')
    depth_metric, _, height, params = get_height(elem, json_file)#, sky_value=-100, water_height=0.45)
    height = set_zero_reference(height)
    im = Image.fromarray(256*(normalize(np.log(depth_metric/far)))).convert('L')
    im.save(os.path.join(depth_metric_folder, im_id + '.png'))
    np.save(os.path.join(heights_folder, im_id + '.npy'), height)
    with open(os.path.join(cam_params_folder, im_id + '.json'), 'w') as f:
        json.dumps(params)
    f.close()
    print("saved "+ str(im_id))