import cv2
import os
import multi_process_utils as m_utils
import generate_GAN_data


def main():
    #target_dir = '/home/kyle/Documents/senior_project/stylegan/GAN_imgs/'
    new_size = 256
    save_dir = os.path.join(os.getcwd(), 'GAN_images')
    #latent_list = np.load(os.path.join(save_dir, 'latents_list.npy'))
    total_num_imgs = 19980
    paths = []
    for k in range(total_num_imgs):
        paths.append(os.path.join(save_dir, str(k) + '.png'))
    print("Loading Images in")
    images = m_utils.multi_process(m_utils.get_img_256, paths, os.cpu_count())
    new_img_paths = []
    new_save_dir = os.path.join(os.getcwd(), 'GAN_images_256')
    os.makedirs(new_save_dir, exist_ok=True)
    for j in range(total_num_imgs):
        new_img_paths.append(os.path.join(new_save_dir, str(j) + '.png'))
    print("Saving Images")
    m_utils.multi_thread(generate_GAN_data.save_img, zip(images, new_img_paths), os.cpu_count())



if __name__ == "__main__":
    main()