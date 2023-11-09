from imageGenerator import get_images_from_diffusion
from poses.pose_utils import gen_poses

if __name__ == '__main__':
    imageList = get_images_from_diffusion('A yellow toy lego crane')
    # Matchers that can be used in 'gen_poses' exhaustive_matcher sequential_matcher
    gen_poses('/image_list', 'exhaustive_matcher')
