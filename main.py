from argparse import ArgumentParser
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import libemg

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
        online_model = libemg.emg_predictor.OnlineEMGClassifier(offline_model, window_size, window_increment, online_data_handler, feature_list)
        online_model.run(block=False)

        controller = libemg.environments.controllers.ClassifierController(output_format='predictions', num_classes=len(gesture_ids))
        while True:
            data = controller.get_data(['predictions', 'pc'])
            if data is None:
                continue

            predictions, pc = data
            print(predictions, pc)
            
            if args.action == 'notes':
                ...
            elif args.action == 'chords':
                ...
            else:
                raise ValueError(f"Unexpected value for action. Got: {args.action}.")

    print('Main script complete!')

if __name__ == '__main__':
    main()
