import torch
from diffusers import DDPMParallelScheduler
from diffusers import StableDiffusionParadigmsPipeline
import os
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


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

    # Convert PIL images to tensors
    to_tensor = ToTensor()
    image_list = [to_tensor(image) for image in image_list]

    # Create the output directory if it doesn't exist
    os.makedirs('scenedir/images', exist_ok=True)

    # Save the images to the output directory
    for i, image in enumerate(image_list):
        image_path = os.path.join('scenedir/images', f"image_{i:04d}.png")
        save_image(image, image_path)

    print('[INFO] Number of images created: ', len(image_list))
    print('[INFO] Saving images...')

    return image_list
