import os
import glob
def concatenate_files(directory1, directory2, directory3):
    # Use glob to get a list of files in each directory
    files_dir1 = []
    files_dir2 = []
    files_dir3 = glob.glob(directory3+ '/*.npy')


    # Concatenate content of each file with a space
    concatenated_content = []

    for root, dirs, files in os.walk(directory1):
        for file_name in files:
            if file_name.endswith('.png'):  # Adjust the extension as needed
                files_dir1.append(os.path.join(root, file_name))

    for root, dirs, files in os.walk(directory2):
        for file_name in files:
            if file_name.endswith(
                    '.png'):  # Adjust the extension as needed
                files_dir2.append(os.path.join(root, file_name))

    files_dir1.sort()
    files_dir2.sort()
    files_dir3.sort()
    pred_length = len(files_dir3)
    print('lengths', len(files_dir1), len(files_dir2), len(files_dir3))
    #assert len(files_dir1) == len(files_dir2) == len(files_dir3)


    with open('output.txt', 'w') as file:
        for i in range(len(files_dir3)):

            path1 = files_dir1[i]
            path2 = files_dir2[i]
            path3 = files_dir3[i]
            # Open a text file for writing
            file.write(f'{path1} {path2} {path3}\n')

    return concatenated_content

# dir1 = input
# dir2 = gt
# dir3 = prediction

# Replace 'path/to/dir1' and 'path/to/dir2' with your actual directory paths
#directory1 = '/home/petu/datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train'
#directory2 = '/home/petu/datasets/CityScapes/disparity_trainvaltest/disparity/train'
#directory3 = '/home/petu/projects/claire/ZoeDepth/results/citi_out'
directory1 = '/Users/clairelin/Documents/research/datasets/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train'
directory2 = '/Users/clairelin/Documents/research/datasets/CityScapes/disparity_trainvaltest/disparity/train'
directory3 = '/Users/clairelin/Documents/research/vidar/results/citi_out'
#directory3 = '/home/petu/projects/claire/Marigold/results/citi_out/depth_npy'
# directory1 = '/home/petu/datasets/kitti/kitti_test_inputs'
# directory2 = '/home/petu/datasets/kitti/kitti_test_gts'
# directory3 = '/home/petu/projects/claire/ZoeDepth/results/kitti_out_run_image_nk'
#directory3 = '/Users/clairelin/Documents/research/Marigold/results/kitti_out/depth_npy'

result_list = concatenate_files(directory1, directory2, directory3)

# Now result_list contains the concatenated content of each file with a space
print(result_list)
