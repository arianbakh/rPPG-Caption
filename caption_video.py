import argparse
import bisect
import cv2
import math
import numpy as np
import os
import shutil
import subprocess
import zipfile

from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip


def run_command(command, conda_dir, env_name):
    activation_command = f'source {conda_dir}/etc/profile.d/conda.sh && conda activate {env_name} && {command}'    
    process = subprocess.Popen(activation_command, shell=True, executable="/bin/bash")
    process.communicate()


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count


def create_pulse_video(
    pulse_time_series,
    clip_height,
    fps,
    pulse_video_path,
    limit_seconds=6,
    padding_pixels=5,
    width_coeff=3,
    color=(77, 71, 255)
):
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
        last_x = None
        last_y = None
        for j in range(len(xdata) - 1):
            x1 = int(xdata[j] / limit_data_points * clip_width)
            y1 = clip_height - int(ydata[j])
            x2 = int(xdata[j + 1] / limit_data_points * clip_width)
            y2 = clip_height - int(ydata[j + 1])
            last_x = x2
            last_y = y2
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        if last_x is not None and last_y is not None:
            cv2.circle(frame, (x2, y2), 5, color, -1)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def get_bounds(mix_dir, mix_video_name):
    _, _, _, frame_count = get_video_info(os.path.join(mix_dir, mix_video_name))
    start_frame = int(mix_video_name.split('_')[0].split('-')[1][1:])
    return start_frame, frame_count


def get_clip_data_structures(mix_dir):
    mix_video_names = list(os.listdir(mix_dir))
    mix_locations = {}
    events = []
    for mix_video_name in mix_video_names:
        start_frame, frame_count = get_bounds(mix_dir, mix_video_name)
        events.append({
            'time': start_frame,
            'mix_video_name': mix_video_name,
            'type': 'begin',
        })
        events.append({
            'time': start_frame + frame_count - 1,
            'mix_video_name': mix_video_name,
            'type': 'end',
        })
        mix_locations[mix_video_name] = -1  # TODO error handling: check none are -1 in the end
    available_locations = list(range(len(mix_video_names)))
    sorted_events = sorted(events, key=lambda x: (x['time'], 0 if x['type'] == 'end' else 1))
    for event in sorted_events:
        if event['type'] == 'end':
            mix_location = mix_locations[event['mix_video_name']]
            assert mix_location >= 0  # a clip should begin before it ends
            bisect.insort_left(available_locations, mix_location)
        elif event['type'] == 'begin':
            mix_location = available_locations[0]
            available_locations.remove(mix_location)
            mix_locations[event['mix_video_name']] = mix_location
    max_concurrent_clips = max(mix_locations.values())
    return max_concurrent_clips, mix_locations


def run(args):  # TODO break into functions
    video_file_name = os.path.basename(args.video_path).split('.')[0]
    abs_video_path = os.path.abspath(args.video_path)
    video_fps, video_width, video_height, _ = get_video_info(abs_video_path)

    # calculate clip size and location
    if video_width >= 1024:
        unit_size = video_width // 16
        columns = 4
    elif 512 <= video_width < 1024:
        unit_size = video_width // 8
        columns = 2
    else:
        unit_size = video_width // 4
        columns = 1

    # get clips
    abs_fd_dir = os.path.abspath(args.fd_dir)
    abs_output_dir = os.path.abspath(args.output_dir)
    face_video_command = 'python %s/face_video.py --video-path %s --output-dir %s --num-processes %d' % (
        abs_fd_dir,
        abs_video_path,
        abs_output_dir,
        args.num_processes
    )
    abs_conda_dir = os.path.abspath(args.conda_dir)
    run_command(face_video_command, abs_conda_dir, 'fd')
    zip_file_path = os.path.join(abs_output_dir, '%s_clips.zip' % video_file_name)
    clips_dir = os.path.join(abs_output_dir, '%s_clips' % video_file_name)
    if os.path.exists(clips_dir):
        shutil.rmtree(clips_dir)
    os.makedirs(clips_dir)
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
    mix_dir = os.path.join(abs_output_dir, '%s_mix' % video_file_name)
    if not os.path.exists(mix_dir):
        os.makedirs(mix_dir)
    for clip_file_name in os.listdir(clips_dir):
        clip_path = os.path.join(clips_dir, clip_file_name)
        clip_fps, _, _, _ = get_video_info(clip_path)

        # get rPPG for clip
        pulse_path = os.path.join(pulse_dir, clip_file_name.split('.')[0] + '.npy')
        rppg_command = 'CUDA_VISIBLE_DEVICES= python %s/code/predict_vitals.py --video_path %s --pulse-dir %s --model-path %s' % (
            abs_mtts_can_dir,
            clip_path,
            pulse_path,
            os.path.join(abs_mtts_can_dir, 'mtts_can.hdf5')
        )
        run_command(rppg_command, abs_conda_dir, 'tf-gpu')

        # create pulse video for clip
        with open(pulse_path, 'rb') as pulse_file:
            pulse_time_series = np.load(pulse_file)
        pulse_video_path = os.path.join(pulse_video_dir, clip_file_name.split('.')[0] + '_pulse.mp4')
        create_pulse_video(pulse_time_series, unit_size, clip_fps, pulse_video_path)

        # mix clip and pulse video
        clip = VideoFileClip(clip_path).resize(height=unit_size)
        pulse_video = VideoFileClip(pulse_video_path)
        mix_clip = clips_array([[clip, pulse_video]])
        mix_path = os.path.join(mix_dir, clip_file_name.split('.')[0] + '_mix.mp4')
        mix_clip.write_videofile(mix_path)

    # create final video
    max_concurrent_clips, mix_locations = get_clip_data_structures(mix_dir)
    original_video = VideoFileClip(abs_video_path)
    videos_to_compose = []
    padded_video = original_video.margin(bottom=unit_size * int(math.ceil(max_concurrent_clips / columns)))
    videos_to_compose.append(padded_video)
    for mix_video_name, mix_location in mix_locations.items():
        mix_row = mix_location // columns
        mix_col = mix_location % columns
        start_frame, _ = get_bounds(mix_dir, mix_video_name)
        augmented_mix_clip = VideoFileClip(
            os.path.join(mix_dir, mix_video_name)
        ).set_start(
            start_frame / video_fps
        ).set_position(
            (
                mix_col * 4 * unit_size,  # 1 + width_coeff = 4
                video_height + mix_row * unit_size
            )
        )
        videos_to_compose.append(augmented_mix_clip)
    final_video = CompositeVideoClip(videos_to_compose)
    final_video.write_videofile(os.path.join(abs_output_dir, '%s_final.mp4' % video_file_name))


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
