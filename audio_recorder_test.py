import os
import time
import pyaudio
import wave
import cv2
import torch
import transforms
from model import generate_model
from torch.autograd import Variable
import zmq
import numpy as np
import librosa
import soundfile as sf
from facenet_pytorch import MTCNN
from tqdm import tqdm
import functools
from PIL import Image
from opts import parse_opts


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")
FLAG = 1

# from test import audio_extractor, faces_extractor, load_audio, get_default_video_loader, parse_opts, get_mfccs, server

def audio_extractor(root='RAVDESS/'):
    target_time = 3.6  # sec
    for actor in os.listdir(root):
        for audiofile in os.listdir(os.path.join(root, actor)):

            if not audiofile.endswith('.wav') or 'croppad' in audiofile:
                continue

            audios = librosa.core.load(os.path.join(root, actor, audiofile), sr=22050)

            y = audios[0]  # numpy type save
            sr = audios[1]
            target_length = int(sr * target_time)
            if len(y) < target_length:
                y = np.array(list(y) + [0 for i in range(target_length - len(y))])
            else:
                remain = len(y) - target_length
                y = y[remain // 2:-(remain - remain // 2)]

            sf.write(os.path.join(root, actor, audiofile[:-4] + '_croppad.wav'), y, sr)

def faces_extractor(root='RAVDESS/'):
    mtcnn = MTCNN(image_size=(720, 1280), device=device)

    # mtcnn.to(device)
    save_frames = 15
    input_fps = 30

    save_length = 3.6  # seconds
    save_avi = True

    failed_videos = []

    select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    n_processed = 0
    for sess in tqdm(sorted(os.listdir(root))):
        for filename in os.listdir(os.path.join(root, sess)):
            if filename.endswith('.mp4'):

                cap = cv2.VideoCapture(os.path.join(root, sess, filename))
                # calculate length in frames
                framen = 0
                while True:
                    i, q = cap.read()
                    if not i:
                        break
                    framen += 1
                cap = cv2.VideoCapture(os.path.join(root, sess, filename))

                if save_length * input_fps > framen:
                    skip_begin = int((framen - (save_length * input_fps)) // 2)
                    for i in range(skip_begin):
                        _, im = cap.read()

                framen = int(save_length * input_fps)
                frames_to_select = select_distributed(save_frames, framen)
                save_fps = save_frames // (framen // input_fps)
                if save_avi:
                    out = cv2.VideoWriter(os.path.join(root, sess, filename[:-4] + '_facecroppad.avi'),
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

                numpy_video = []
                success = 0
                frame_ctr = 0

                while True:
                    ret, im = cap.read()
                    if not ret:
                        break
                    if frame_ctr not in frames_to_select:
                        frame_ctr += 1
                        continue
                    else:
                        frames_to_select.remove(frame_ctr)
                        frame_ctr += 1

                    try:
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    except:
                        failed_videos.append((sess, i))
                        break

                    temp = im[:, :, -1]
                    im_rgb = im.copy()
                    im_rgb[:, :, -1] = im_rgb[:, :, 0]
                    im_rgb[:, :, 0] = temp
                    im_rgb = torch.tensor(im_rgb)
                    im_rgb = im_rgb.to(device)

                    bbox = mtcnn.detect(im_rgb)
                    if bbox[0] is not None:
                        bbox = bbox[0][0]
                        bbox = [round(x) for x in bbox]
                        x1, y1, x2, y2 = bbox
                    im = im[y1:y2, x1:x2, :]
                    im = cv2.resize(im, (224, 224))
                    if save_avi:
                        out.write(im)
                    numpy_video.append(im)
                if len(frames_to_select) > 0:
                    for i in range(len(frames_to_select)):
                        if save_avi:
                            out.write(np.zeros((224, 224, 3), dtype=np.uint8))
                        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                if save_avi:
                    out.release()
                np.save(os.path.join(root, sess, filename[:-4] + '_facecroppad.npy'), np.array(numpy_video))
                if len(numpy_video) != 15:
                    print('Error', sess, filename)

        n_processed += 1
        with open('processed.txt', 'a') as f:
            f.write(sess + '\n')
        print(failed_videos)

def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))
    # print("video data: ", video_data)
    return video_data  # list of PIL.Image.Image, PIL.Image.open读入的是RGB顺序，而opencv中cv2.imread读入的是BGR通道顺序

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def audio_recorder(wave_output_filename='output.wav'):
    buffer = []
    t = time.time()
    duration = 4
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, frames_per_buffer=2048)
    print('**********Recording Start**********')
    while True:
        if time.time() < t + duration:
            # each loop read 2048 frames
            string_audio_data = stream.read(2048)
            buffer.append(string_audio_data)
        else:
            print('**********Recording end THIS TIME**********')
            wf = wave.open(f"{time.time()}-{wave_output_filename}", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes("".encode().join(buffer))
            wf.close()
            stream.stop_stream()
            stream.close()
            pa.terminate()
            buffer = []
            print('**********Recording Start**********')
            # prepare for entering next loop
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, frames_per_buffer=2048)
            t = time.time()
            continue


def video_recorder(video_output_filename='output.mp4'):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{time.time()}-{video_output_filename}", fourcc, 20, (width, height))

    t = time.time()
    duration = 4

    print('**********Recording Start**********')
    while True:
        ret, frame = cap.read()
        cv2.imshow('Preview_Window', frame)
        if time.time() < t + duration:

            if ret:
                out.write(frame)
        else:
            out.release()
            print('**********Recording end THIS TIME**********')


            out = cv2.VideoWriter(f"{time.time()}-{video_output_filename}", fourcc, 20, (width, height))
            print('**********Recording Start**********')
            t = time.time()
            continue

        if cv2.waitKey(5) == ord('q'):  # 等到5ms检测是否退出，同时防止读取太快
            #cv2.destroyWindow('Preview_Window')
            break


def audio_and_video_recorder(model, opt, output_filename='RAVDESS'):
    i = 1
    # Video Initialization
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{output_filename}/video/test.mp4", fourcc, 20, (width, height))

    # Audio Initialization
    buffer = []
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, frames_per_buffer=2048)

    # time / duration initialization
    t = time.time()
    duration = 3.6

    print('**********Recording Start**********')
    while True:
        ret, frame = cap.read()
        cv2.imshow('Preview_Window', frame)
        if time.time() < t + duration:
            # audio record
            string_audio_data = stream.read(2048)
            buffer.append(string_audio_data)
            # video record
            if ret:
                out.write(frame)
        else:
            out.release()
            print('**********Recording end THIS TIME**********')
            wf = wave.open(f"{output_filename}/audio/test.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes("".encode().join(buffer))
            wf.close()
            stream.stop_stream()
            stream.close()
            pa.terminate()
            buffer = []

            print('**********Model Processing**********')
            audio_extractor()
            faces_extractor()

            video_transform = transforms.Compose([transforms.ToTensor(opt.video_norm_value)])
            spatial_transform = video_transform
            loader = get_default_video_loader()
            # video_path = 'RAVDESS/video/02-02-06-02-02-02-01_facecroppad.npy'
            ori_video_path = 'RAVDESS/video/test.mp4'
            video_path = 'RAVDESS/video/test_facecroppad.npy'
            clip = loader(video_path)
            if video_transform is not None:
                spatial_transform.randomize_parameters()
                clip = [spatial_transform(img) for img in clip]  # list of transformed PIL
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            # audio_path = 'RAVDESS/audio/03-02-06-02-02-02-01_croppad.wav'
            audio_path = 'RAVDESS/audio/test_croppad.wav'
            ori_audio_path = 'RAVDESS/audio/test.wav'
            audio_transform = None
            y, sr = load_audio(audio_path, sr=22050)
            if audio_transform is not None:
                audio_transform.randomize_parameters()
                y = audio_transform(y)

            mfcc = get_mfccs(y, sr)
            audio_features = mfcc

            audio_inputs = torch.tensor(audio_features)
            visual_inputs = torch.tensor(clip)

            visual_inputs = visual_inputs.permute(1, 0, 2, 3)
            with torch.no_grad():
                audio_inputs = Variable(audio_inputs).unsqueeze(0)  # 需要加上Batch size
                visual_inputs = Variable(visual_inputs)
            outputs = model(audio_inputs, visual_inputs)
            # print(outputs, torch.nn.Softmax(dim=1)(outputs))
            outputs = torch.nn.Softmax(dim=1)(outputs)
            os.remove(video_path)
            os.remove(audio_path)
            os.remove(ori_audio_path)
            os.remove(ori_video_path)


            outputs = outputs[0][1:]  # clam, happy, sad, angry, fearful, disgust, surprised
            # print(outputs)
            print('Serial number of data: ' + str(i))
            print('send message...' + str(FLAG) + str(outputs.tolist()))

            socket.send_string(str(FLAG) + str(outputs.tolist()))
            i = i + 1
            print('**********Sending Over**********')

            print('**********Recording Start**********')
            t = time.time()

            # video record preparation for next time
            out = cv2.VideoWriter(f"{output_filename}/video/test.mp4", fourcc, 20, (width, height))
            # audio record preparation for next time
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True, frames_per_buffer=2048)
            continue

        if cv2.waitKey(5) == ord('q'):  # 等到5ms检测是否退出，同时防止读取太快
            #cv2.destroyWindow('Preview_Window')
            break


if __name__ == '__main__':
    # audio_recorder()
    # video_recorder()
    opt = parse_opts()
    model, parameters = generate_model(opt)
    model.eval()
    resume_path = 'pre-trained_models/RAVDESS_multimodalcnn_15_checkpoint0_20230203.pth'
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], False)
    with torch.no_grad():
        audio_and_video_recorder(model=model, opt=opt,output_filename='RAVDESS')