import os
import torch
from torchvision.transforms import ToTensor, functional as F
from torchvision.utils import save_image
from diffusers import DDPMParallelScheduler, StableDiffusionParadigmsPipeline


def get_images_from_diffusion(prompt):
    scheduler = DDPMParallelScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",
                                                      timestep_spacing="trailing")
    pipe = StableDiffusionParadigmsPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", scheduler=scheduler, torch_dtype=torch.float16, height=100, width=100
    )
    pipe = pipe.to("cuda")
    ngpu, batch_per_device = torch.cuda.device_count(), 5
    pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])

    image_list = pipe(prompt, parallel=ngpu * batch_per_device, num_inference_steps=1000).images

    # Create the output directory if it doesn't exist
    output_dir = 'scenedir/images'
    os.makedirs(output_dir)

    # Save the images to the output directory
    for i, image in enumerate(image_list):
        # Convert the PIL image to a PyTorch tensor before saving
        image_tensor = ToTensor()(image)
        # Save the tensor as an image file
        image_path = os.path.join(output_dir, f"image_{i:04d}.png")
        save_image(image_tensor, image_path)

    print('[INFO] Number of images created: ', len(image_list))
    print('[INFO] Saving images to:', output_dir)

    return image_list