import os, time, h5py
import numpy as np
from lstm_btc import lstm, etl, plotting, config, conn

tstart = time.time()
dl = etl.ETL()
true_values = []
all_predictions = []
holdings, wallet = 100, 100


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


def update_wallet(xs, ys):
    """Running talley of our experiment (how much we'll make)"""
    global wallet, holdings
    for x, y in zip(xs, ys):
        x = float(x)
        if x < 0:
            wallet += holdings
            holdings = 0
        elif x > 0:
            holdings += wallet
            wallet = 0
        holdings += holdings*(.1*y)
    print('holdings=${} wallet=${}'.format(holdings, wallet))


dl.ensure_data()

if config.flags.create_clean_data or not os.path.isfile(config.data.filename_clean):
    print('> Generating clean data from:', config.data.filename_clean, 'with batch_size:',
          config.data.batch_size)
    dl.create_clean_datafile()

if config.flags.train or not os.path.isfile(config.model.filename_model):
    data_gen_train = dl.generate_clean_data()

    with h5py.File(config.data.filename_clean, 'r') as hf:
        nrows = hf['x'].shape[0]
        ncols = hf['x'].shape[2]

    ntrain = int(config.data.train_test_split * nrows)
    steps_per_epoch = int((ntrain / config.model.epochs) / config.data.batch_size)
    print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

    model = lstm.build_network([ncols, 150, 150, 1])
    model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=config.model.epochs
    )
    model.save(config.model.filename_model)
    print('> Model Trained! Weights saved in', config.model.filename_model)

    data_gen_test = dl.generate_clean_data(start_index=ntrain)

    ntest = nrows - ntrain
    steps_test = int(ntest / config.data.batch_size)
    print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

    predictions = model.predict_generator(
        dl.generator_strip_xy(data_gen_test, true_values),
        steps=steps_test
    )

    # Save our predictions
    with h5py.File(config.model.filename_predictions, 'w') as hf:
        dset_p = hf.create_dataset('predictions', data=predictions)
        dset_y = hf.create_dataset('true_values', data=true_values)

    update_wallet(predictions, true_values)
    plotting.plot_results(predictions[-800:], true_values[-800:], block=True)

else:
    # Initialize an array of 50 points, since current matplotlib code won't work with dynamic window-sizes. We'll
    # just modify this window inline and re-draw
    true_values = [0] * 50
    all_predictions = [0] * 50

    while True:
        dl.fetch_market_and_save()
        data_gen_test = dl.data_tail()
        model = lstm.load_network()

        x, y = data_gen_test
        true_values.pop(0);true_values.append(y)
        predictions = model.predict_on_batch(x)
        all_predictions.pop(0);all_predictions.append(predictions[0])

        update_wallet(predictions, [y])
        print('prediction:', all_predictions[-1], 'true:', true_values[-1])
        plotting.plot_results(all_predictions, true_values)


if config.flags.multi_window:
    # Reload the data-generator
    data_gen_test = dl.generate_clean_data(
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
