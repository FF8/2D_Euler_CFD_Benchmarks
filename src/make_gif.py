# make_gif.py
import imageio.v2 as imageio
import os
import glob

def create_gif(image_folder='output_frames', gif_name='simulation.gif', fps=15):
    """
    Creates a GIF from a folder of PNG images.
    """
    print(f"Searching for frames in '{image_folder}/'...")
    
    # Find all .png files and sort them numerically
    search_path = os.path.join(image_folder, 'frame_*.png')
    filenames = sorted(glob.glob(search_path))

    if not filenames:
        print(f"Error: No frames found in '{image_folder}'. Did you run the simulation with --save_frames?")
        return

    print(f"Found {len(filenames)} frames. Creating GIF...")
    
    # The 'loop=0' argument makes the GIF loop indefinitely
    with imageio.get_writer(gif_name, mode='I', fps=fps, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f"Successfully created '{gif_name}'")

if __name__ == '__main__':
    # You can customize the folder, output name, and frames per second here
    create_gif(image_folder='output_frames', gif_name='sedov_explosion_simulation.gif', fps=20)
