from argparse import ArgumentParser
from pathlib import Path
from collections import deque
# from multiprocessing import Process, Lock
from threading import Thread, Lock

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import libemg
import pyaudio
import wave


# def play_audio(model_data, previous_prediction, wf):
#     # with wave.open('audio-files/A4.wav') as wf:
#     def callback(in_data, frame_count, time_info, status):
#         data = wf.readframes(frame_count)
#         prediction, pc = model_data
#         prediction = prediction[0]  # convert from 1-element list
#         pc = pc[0]  # convert from 1-element list
#         flag = pyaudio.paAbort if previous_prediction != prediction else pyaudio.paContinue
#         return (data, flag)

                # if wf[0] is not None:
                #     wf[0].close()
                # # Load in new waveform file
                # wav_file = class_to_audio_map[prediction]
                # if wav_file is None:
                #     wf[0] = None
                #     continue

                # wf[0] = wave.open(f"audio-files/{wav_file}", 'rb')
        
def play_audio(waveform_filename, stop_flag, lock):
    sample_wav_file = wave.open('audio-files/A4.wav', 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_wav_file.getsampwidth()),
                channels=sample_wav_file.getnchannels(),
                rate=sample_wav_file.getframerate(),
                output=True)

    current_waveform_filename = waveform_filename
    wf = None
    chunk = 1024
    while True:
        with lock:
            wvf = waveform_filename[0]
        # print(waveform_filename)
        # print(wvf)
        if wvf is None:
            wf = None
            continue
        
        if wvf != current_waveform_filename:
            if wf is not None:
                wf.close()
            wf = wave.open(f"audio-files/{wvf}", 'rb')

        current_waveform_filename = wvf

        if wf is None:
            continue
        
        # print('hi')
        wav_data = wf.readframes(chunk)
        while wav_data and not stop_flag[0]:
            stream.write(wav_data)
            wav_data = wf.readframes(chunk)
            # print(wav_data)
    # print('done')


# def handle_audio(wf, stream):
#     while




def main():
    parser = ArgumentParser(prog='EMG Music', description='Play a song using LibEMG!')
    parser.add_argument('subject', help='Subject ID.')
    parser.add_argument('action', choices=('sgt', 'notes', 'chords'), help='Action to perform.')
    args = parser.parse_args()
    print(args)

    data_path = Path('data', args.subject).absolute().as_posix()
    _, smi = libemg.streamers.myo_streamer()
    online_data_handler = libemg.data_handler.OnlineDataHandler(smi)
    gui_args = {
        'media_folder': 'images/',
        'data_folder': data_path,
        'num_reps': 5,
        'rep_time': 1,
        'rest_time': 1
    }
    gesture_ids = [1, 2, 3, 4, 5, 6, 7]
    if args.action == 'sgt':
        gui = libemg.gui.GUI(online_data_handler, args=gui_args, clean_up_on_kill=True)
        gui.download_gestures(gesture_ids, folder=gui_args['media_folder'])
        gui.start_gui()
    else:
        window_size = 20
        window_increment = 10
        odh = libemg.data_handler.OfflineDataHandler()
        regex_filters = [
            libemg.data_handler.RegexFilter(left_bound=f"{data_path}/C_", right_bound='_R', values=[str(idx) for idx in range(len(gesture_ids))], description='classes'),
            libemg.data_handler.RegexFilter(left_bound='_R_', right_bound='_emg.csv', values=[str(idx) for idx in range(gui_args['num_reps'])], description='reps')
        ]
        odh.get_data(data_path, regex_filters)
        windows, metadata = odh.parse_windows(window_size=window_size, window_increment=window_increment)
        labels = metadata['classes']
        fe = libemg.feature_extractor.FeatureExtractor()
        feature_list = fe.get_feature_groups()['LS4']
        feature_matrix = fe.extract_features(feature_list, windows, array=True)
        clf = LinearDiscriminantAnalysis()
        clf.fit(feature_matrix, labels)
        offline_model = libemg.emg_predictor.EMGClassifier(clf)
        offline_model.add_velocity(windows, labels)
        offline_model.add_rejection()
        online_model = libemg.emg_predictor.OnlineEMGClassifier(offline_model, window_size, window_increment, online_data_handler, feature_list)
        online_model.run(block=False)

        controller = libemg.environments.controllers.ClassifierController(output_format='predictions', num_classes=len(gesture_ids))
        
        q = deque(maxlen=5)
        current_class = None
        # wf = None
        class_to_audio_map = {
            0: 'A4.wav',    # close
            1: 'C4.wav',    # open
            2: None,    # nm
            3: 'D4.wav',    # pronation
            4: 'E4.wav',    # supination
            5: 'F4.wav',    # extension
            6: 'G4.wav' # flexion
        }
        # def callback(in_data, frame_count, time_info, status):
        #     data = wf.readframes(frame_count)
        #     if prediction
            
        # def callback(in_data, frame_count, time_info, status):
        #     wav_data = wf.readframes(frame_count)
        #     prediction, pc = data
        #     prediction = prediction[0]  # convert from 1-element list
        #     pc = pc[0]  # convert from 1-element list
        #     flag = pyaudio.paAbort if current_class != prediction else pyaudio.paContinue
        #     return (wav_data, flag)
        sample_wav_file = wave.open('audio-files/A4.wav', 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sample_wav_file.getsampwidth()),
                    channels=sample_wav_file.getnchannels(),
                    rate=sample_wav_file.getframerate(),
                    output=True)
        stream = [stream]
        # wf = [None]
        waveform_filename = [None]
        stop_flag = [False]
        lock = Lock()
        thread = Thread(target=play_audio, args=(waveform_filename, stop_flag, lock))
        thread.start()
        # process = Process(target=play_audio, args=(waveform_filename, stop_flag, lock))
        # process.start()

        try:
            while True:
                data = controller.get_data(['predictions', 'pc'])
                if data is None:
                    continue

                prediction, pc = data
                prediction = prediction[0]  # convert from 1-element list
                pc = pc[0]  # convert from 1-element list
                print(prediction, pc)

                q.append(prediction)
                with lock:
                    # print('hi')
                    if not all([p == prediction for p in q]) or prediction == -1:
                        # play_audio(wf, stream)
                        stop_flag = [False]
                        continue

                    # print('hi again')
                    stop_flag = [True]
                    waveform_filename[0] = class_to_audio_map[prediction]
                    # print(waveform_filename)
                
                if args.action == 'notes':
                    ...
                elif args.action == 'chords':
                    ...
                else:
                    raise ValueError(f"Unexpected value for action. Got: {args.action}.")
        except KeyboardInterrupt:
            thread.join()
            # process.join()
    print('Main script complete!')

if __name__ == '__main__':
    main()
