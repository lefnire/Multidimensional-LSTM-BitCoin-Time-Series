import os, time, threading, json, h5py, argparse, itertools
import numpy as np
import pandas as pd
import lstm, etl, plotting

configs = json.loads(open('configs.json').read())
tstart = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--clean-data', help='Should create & save clean dataset? (~7 minutes)', action='store_true')
parser.add_argument('--train', help='Should train the RNN? (~10 minutes)', action='store_true')
parser.add_argument('--multi-window', help='Display prediction over a larger window?', action='store_true')
args = vars(parser.parse_args())
should_create_clean_data = args['clean_data'] or not os.path.isfile(configs['data']['filename_clean'])
should_train = args['train'] or not configs['model']['filename_model']
multi_window = args['multi_window']


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

true_values = []
all_predictions = []
holdings, wallet = 100, 100

def generator_strip_xy(data_gen, true_values):
    for x, y in data_gen:
        true_values += list(y)
        yield x


def fit_model_threaded(model, data_gen_train, steps_per_epoch, configs):
    """thread worker for model fitting - so it doesn't freeze on jupyter notebook"""
    # model = lstm.build_network([ncols, 150, 150, 1])
    model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=configs['model']['epochs']
    )
    model.save(configs['model']['filename_model'])
    print('> Model Trained! Weights saved in', configs['model']['filename_model'])
    return


dl = etl.ETL()

if should_train:
    if should_create_clean_data:
        dl.create_clean_datafile(
            filename_in=configs['data']['filename'],
            filename_out=configs['data']['filename_clean'],
            batch_size=configs['data']['batch_size'],
            x_window_size=configs['data']['x_window_size'],
            y_window_size=configs['data']['y_window_size'],
            y_col=configs['data']['y_predict_column'],
            filter_cols=configs['data']['filter_columns'],
            normalise=True
        )

    print('> Generating clean data from:', configs['data']['filename_clean'], 'with batch_size:',
          configs['data']['batch_size'])

    data_gen_train = dl.generate_clean_data(
        configs['data']['filename_clean'],
        batch_size=configs['data']['batch_size']
    )

    with h5py.File(configs['data']['filename_clean'], 'r') as hf:
        nrows = hf['x'].shape[0]
        ncols = hf['x'].shape[2]

    ntrain = int(configs['data']['train_test_split'] * nrows)
    steps_per_epoch = int((ntrain / configs['model']['epochs']) / configs['data']['batch_size'])
    print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

    model = lstm.build_network([ncols, 150, 150, 1])
    # FIXME issues with threading, see https://github.com/jaungiers/Multidimensional-LSTM-BitCoin-Time-Series/issues/1
    # t = threading.Thread(target=fit_model_threaded, args=[model, data_gen_train, steps_per_epoch, configs])
    # t.start()
    fit_model_threaded(model, data_gen_train, steps_per_epoch, configs)

    data_gen_test = dl.generate_clean_data(
        configs['data']['filename_clean'],
        batch_size=configs['data']['batch_size'],
        start_index=ntrain
    )

    ntest = nrows - ntrain
    steps_test = int(ntest / configs['data']['batch_size'])
    print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

    predictions = model.predict_generator(
        generator_strip_xy(data_gen_test, true_values),
        steps=steps_test
    )

    # Save our predictions
    with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
        dset_p = hf.create_dataset('predictions', data=predictions)
        dset_y = hf.create_dataset('true_values', data=true_values)


    # Running talley of our experiment (how much we'll make)
    holdings, wallet = 100, 100
    for x, y in zip(predictions, true_values):
        x = float(x)
        if x < 0:
            wallet += holdings
            holdings = 0
        elif x > 0:
            holdings += wallet
            wallet = 0
        holdings += holdings*(.1*y)
    print('holdings=${} wallet=${}'.format(holdings, wallet))

    plotting.plot_results(predictions[-800:], true_values[-800:], block=True)

else:
    true_values = [0] * 50
    all_predictions = [0] * 50
    while True:
        dl.fetch_market_and_save()
        print('> Generating clean data from:', configs['data']['filename_clean'], 'with batch_size:',
              configs['data']['batch_size'])

        data_gen_test = dl.data_tail(batch_size=configs['data']['batch_size'],
            x_window_size=configs['data']['x_window_size'], y_window_size=configs['data']['y_window_size'],
            y_col=configs['data']['y_predict_column'])

        model = lstm.load_network(configs['model']['filename_model'])

        # ntest = dl.nrows
        # # steps_test = int(ntest / configs['data']['batch_size'])
        # steps_test = int(ntest / batch_size)
        # # steps_test = 98  # FIXME what's going on here? ntest / 98 = 110, this is max num allowed else StopIteration error
        # print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

        x, y = data_gen_test
        true_values.pop(0);true_values.append(y)
        predictions = model.predict_on_batch(x)
        all_predictions.pop(0);all_predictions.append(predictions[0])

        # Running talley of our experiment (how much we'll make)
        x = float(all_predictions[-1])
        if x < 0:
            wallet += holdings
            holdings = 0
        elif x > 0:
            holdings += wallet
            wallet = 0
        holdings += holdings * (.1 * y)
        print('holdings=${} wallet=${}'.format(holdings, wallet))

        print('prediction:', all_predictions[-1], 'true:', true_values[-1])
        plotting.plot_results(all_predictions, true_values)


if multi_window:
    # Reload the data-generator
    data_gen_test = dl.generate_clean_data(
        configs['data']['filename_clean'],
        batch_size=800,
        start_index=ntrain
    )
    data_x, true_values = next(data_gen_test)
    window_size = 50  # numer of steps to predict into the future

    # We are going to cheat a bit here and just take the next 400 steps from the testing generator and predict that
    # data in its whole
    predictions_multiple = predict_sequences_multiple(
        model,
        data_x,
        data_x[0].shape[0],
        window_size
    )

    plotting.plot_results_multiple(predictions_multiple, true_values, window_size)
