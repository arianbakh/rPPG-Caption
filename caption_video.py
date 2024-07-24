import argparse
import cv2
import numpy as np
import os
import subprocess
import zipfile


def run_command(command, conda_dir, env_name):
    activation_command = f'source {conda_dir}/etc/profile.d/conda.sh && conda activate {env_name} && {command}'    
    process = subprocess.Popen(activation_command, shell=True, executable="/bin/bash")
    process.communicate()


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height


def create_pulse_video(pulse_time_series, clip_height, fps, pulse_video_path, limit_seconds=3, padding_pixels=5, width_coeff=3):
    limit_data_points = limit_seconds * fps
    normalized_time_series = (
        pulse_time_series - np.min(pulse_time_series)
    ) / (
        np.max(pulse_time_series) - np.min(pulse_time_series)
    ) * (clip_height - 2 * padding_pixels) + padding_pixels
    clip_width = width_coeff * clip_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter(
        pulse_video_path,
        fourcc,
        fps,
        (clip_width, clip_height)
    )
    for i in range(len(pulse_time_series)):
        start = max(0, i - limit_data_points + 1)
        ydata = normalized_time_series[start:i + 1]
        xdata = np.arange(0, len(ydata))
        frame = np.zeros((clip_height, clip_width, 3), dtype=np.uint8)
        for j in range(len(xdata) - 1):
            x1 = int(xdata[j] / limit_data_points * clip_width)
            y1 = clip_height - int(ydata[j])
            x2 = int(xdata[j + 1] / limit_data_points * clip_width)
            y2 = clip_height - int(ydata[j + 1])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def run(args):  # TODO break into functions
    video_file_name = os.path.basename(args.video_path).split('.')[0]

    # get clips
    abs_fd_dir = os.path.abspath(args.fd_dir)
    abs_video_path = os.path.abspath(args.video_path)
    abs_output_dir = os.path.abspath(args.output_dir)
    face_video_command = 'python %s/face_video.py --video-path %s --output-dir %s --num-processes %d' % (
        abs_fd_dir,
        abs_video_path,
        abs_output_dir,
        args.num_processes
    )
    abs_conda_dir = os.path.abspath(args.conda_dir)
    #run_command(face_video_command, abs_conda_dir 'fd')  # TODO uncomment
    zip_file_path = os.path.join(abs_output_dir, '%s_clips.zip' % video_file_name)
    clips_dir = os.path.join(abs_output_dir, '%s_clips' % video_file_name)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(clips_dir)

    # iterate over clips
    abs_mtts_can_dir = os.path.abspath(args.mtts_can_dir)
    pulse_dir = os.path.join(abs_output_dir, '%s_pulses' % video_file_name)
    if not os.path.exists(pulse_dir):
        os.makedirs(pulse_dir)
    pulse_video_dir = os.path.join(abs_output_dir, '%s_pulse_videos' % video_file_name)
    if not os.path.exists(pulse_video_dir):
        os.makedirs(pulse_video_dir)
    for clip_file_name in os.listdir(clips_dir):
        clip_path = os.path.join(clips_dir, clip_file_name)
        clip_fps, clip_width, clip_height = get_video_info(clip_path)

        # get rPPG for clip
        pulse_path = os.path.join(pulse_dir, clip_file_name.split('.')[0] + '.npy')
        rppg_command = 'CUDA_VISIBLE_DEVICES= python %s/code/predict_vitals.py --video_path %s --pulse-dir %s --model-path %s' % (
            abs_mtts_can_dir,
            clip_path,
            pulse_path,
            os.path.join(abs_mtts_can_dir, 'mtts_can.hdf5')
        )
        #run_command(rppg_command, abs_conda_dir, 'tf-gpu')  # TODO uncomment

        # create pulse video for clip
        with open(pulse_path, 'rb') as pulse_file:
            pulse_time_series = np.load(pulse_file)
        pulse_video_path = os.path.join(pulse_video_dir, clip_file_name.split('.')[0] + '_pulse.mp4')
        create_pulse_video(pulse_time_series, clip_height, clip_fps, pulse_video_path)
        # TODO merge clip and pulse video
        # TODO resize merged clip to a standard size
        # TODO merge main video and clips
        # TODO -- requires knowing maximum number of clips per frame
        # TODO -- requires knowing which frames each clip belongs to, add it to the metadata somehow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path of input raw video')
    parser.add_argument('--output-dir', type=str, help='Directory of output faces video')
    parser.add_argument('--fd-dir', type=str, help='rPPG-FaceDetection directory')
    parser.add_argument('--mtts-can-dir', type=str, help='MTTS-CAN directory')
    parser.add_argument('--conda-dir', type=str, help='Anaconda/miniconda directory')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of parallel processes for DNNs (essentially batch size)')
    args = parser.parse_args()
    run(args)
