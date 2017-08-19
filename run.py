import os, time, h5py
import gdax
import numpy as np
from lstm_btc import lstm, etl, plotting, config

tstart = time.time()
dl = etl.ETL()
true_values = []
all_predictions = []
holdings, wallet = 100, 100


def percent_change(new, old):
    if old == 0: return .0001
    return (new - old) / old

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


def buy_sell(preds, trues, tail=False, dry_run=True):
    """Running talley of our experiment (how much we'll make)"""
    global wallet, holdings
    for pred, true in zip(preds, trues):
        if prev == None:
            prev = pred
            continue
        # Just to be clear - this is the anticipated percent change of the change_percent, rather than just change_percent
        # directly. This is because change_percent is normalized in the data processing, so it's not what it seems
        perc_change_of_perc_change = percent_change(pred, prev)
        if perc_change_of_perc_change > .2:
            wallet += holdings
            holdings = 0
        elif perc_change_of_perc_change < -.2:
            holdings += wallet
            wallet = 0
            if not dry_run:
                # Buy 0.01 BTC @ 100 USD
                print("BUYING!")
                # auth_client.buy(price='0.01',  # USD
                #                 # size='0.01', #BTC
                #                 product_id='BTC-USD')
        holdings += holdings*(.1*true)
        prev = pred
    print('holdings=${} wallet=${}'.format(holdings, wallet))


if config.flags.create_clean_data or not os.path.isfile(config.data.filename_clean):
    print('> Generating clean data from:', config.data.filename_clean, 'with batch_size:',
          config.data.batch_size)
    dl.create_clean_datafile(normalize=False)

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

    buy_sell(predictions, true_values)
    plotting.plot_results(predictions, true_values, block=True)

else:
    # Initialize an array of 50 points, since current matplotlib code won't work with dynamic window-sizes. We'll
    # just modify this window inline and re-draw
    true_values = [0.] * 50
    all_predictions = [0.] * 50
    p_changes = [0.] * 50
    t_changes = [0.] * 50
    model = lstm.load_network()

    while True:
        # Market data will be running in the background (populate.py), so each pass will have new data
        x, y = dl.data_tail()

        true = float(y)
        true_values.pop(0);true_values.append(true)
        prediction = float(model.predict_on_batch(x)[0])
        all_predictions.pop(0);all_predictions.append(prediction)

        p_change = percent_change(prediction, all_predictions[-2])
        t_change = percent_change(true, true_values[-2])
        p_changes.pop(0);p_changes.append(p_change)
        t_changes.pop(0);t_changes.append(t_change)

        if p_change >= .05:
            wallet += holdings
            holdings = 0
            print("SELLING!")
        elif p_change <= -.05:
            holdings += wallet
            wallet = 0
            print("BUYING!")
            # # auth_client.buy(price='0.01',  # USD
            # #                 # size='0.01', #BTC
            # #                 product_id='BTC-USD')
        holdings += holdings * (.1 * t_change)
        print('prediction: {} ({}% change); true: {} ({}% change)'.format(prediction, p_change, true, t_change))
        print('holdings=${} wallet=${}'.format(holdings, wallet))

        plotting.plot_results(p_changes, t_changes)
        time.sleep(2)


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
