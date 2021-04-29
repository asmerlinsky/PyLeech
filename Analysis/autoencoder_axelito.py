import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from time import strftime
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
from PyLeech.Utils.unitInfo import UnitInfo
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Sequential
import os

base_dir = "autoencoded_models/"
try:
    os.mkdir(base_dir)
except FileExistsError:
    pass
input_tuple = (  # (N_MODES, N_LAYERS)
    (2, 3),
    (3, 3),
)

N_INPUT_UNITS = 64
LAYER_REDUCTION_FACTOR = 2

cdd = CDU.loadDataDict()

a = [print("%i\t|%s" % (i, list(cdd)[i])) for i in range(len(cdd))]

run_list = []
for fn in list(cdd):
    if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
        pass
    else:
        run_list.append(fn)


for fn in run_list[-5:]:

    ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

    cdd_de3 = cdd[fn]['DE3']
    selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                        neuron_dict["neuron_is_good"]]

    basename = os.path.splitext(os.path.basename(fn))[0]
    try:
        os.mkdir(base_dir + basename)
    except FileExistsError:
        pass

    spike_freq_dict = UnitInfo(fn).spike_freq_dict
    time_length = UnitInfo(fn).time_length

    crawling_interval = cdd[fn]['crawling_intervals']

    bin_step = .5
    sigma = 2
    binned_sfd = burstUtils.processSpikeFreqDict(spike_freq_dict, bin_step, time_interval=crawling_interval,
                                                 time_length=time_length,
                                                 selected_neurons=selected_neurons,
                                                 counting=True)

    # f, a = burstUtils.plotFreq(binned_sfd, scatter_plot=True)
    # f.suptitle(fn)

    sfa = burstUtils.processed_sfd_to_array(binned_sfd)

    smoothed_sfa = burstUtils.processed_sfd_to_array(burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma, 20, bin_step))

    nsamples, nseries = sfa.shape

    # Grafico las series temporales de cada neurona
    fig, ax = plt.subplots(nrows=nseries, ncols=1, figsize=(16, 4 * nseries))
    cmap = get_cmap("tab10")

    for col in range(nseries):
        ax[col].plot(sfa[:, col], color=cmap(col % 10), ls="-", linewidth=.5, marker="o", markersize=1, alpha=.5,
                     label="Crudo")
        ax[col].plot(smoothed_sfa[:, col], color=cmap(col % 10), ls="-", linewidth=1, marker="", markersize=1,
                     label="Suavizado")
        ax[col].set_ylabel(f"Serie no. {col}")
        ax[col].set_xlabel(f"Tiempo [muestras]");
        ax[col].legend(loc="upper left")

        ax[col].minorticks_on()
        ax[col].tick_params(axis='x', which='major', bottom=True)
        ax[col].grid(axis="x", which="major", ls=":")
        ax[col].grid(axis="x", which="minor", ls=":")

    # ax[-1].set_xlim([1000, 1500]);

    fig.tight_layout()
    # fig.savefig(root_dir / ("dataset.png", dpi=200)

    for N_MODES, N_LAYERS in input_tuple:

        # Definimos las capas que del encoder y como se conectan
        encoder_input = Input(shape=(nseries,), name="encoder_input")
        decoder_input = Input(shape=(N_MODES,), name="decoder_input")
        encoder_layers = []
        decoder_layers = []

        # Generamos las capas del encoder de manera programática
        for i in range(N_LAYERS):
            encoder_layers.append(
                Dense(units=N_INPUT_UNITS // LAYER_REDUCTION_FACTOR ** i, activation='relu', name=f"encoder_{i}"))
            decoder_layers.append(
                Dense(units=N_INPUT_UNITS // LAYER_REDUCTION_FACTOR ** (N_LAYERS - 1 - i), activation='relu',
                      name=f"decoder_{i}"))
        encoder_layers.append(Dense(units=N_MODES, activation='linear', name="encoder_output"))
        decoder_layers.append(Dense(units=nseries, activation='linear', name="decoder_output"))

        # Generamos los modelos parciales y el completo
        encoder = Sequential([encoder_input] + encoder_layers, name="encoder")
        decoder = Sequential([decoder_input] + decoder_layers, name="decoder")
        autoencoder = Sequential([encoder_input] + encoder_layers + decoder_layers, name="autoencoder")

        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        timestamp = strftime("%y%m%d_%H%M%S")  # Uso el timestamp para etiquetar este modelo en particular
        tag = f"{timestamp}_N_MODES={N_MODES}_N_LAYERS={N_LAYERS}/"
        output_dir = base_dir + basename + "/" + tag
        os.mkdir(output_dir)

        autoencoder.compile(loss='mae', metrics=['mean_absolute_error'], optimizer='adam')
        # training the model and saving metrics in history
        history = autoencoder.fit(smoothed_sfa, smoothed_sfa,
                                  batch_size=32,
                                  epochs=1000,
                                  validation_split=0.1,
                                  verbose=1)
        autoencoder.save(output_dir + 'modelo_autoencoder.h5')
        with open(output_dir + 'summary_autoencoder.txt', 'w') as f:
            autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('model mean_absolute_error')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(axis="both", ls=":", which="major")

        fig.tight_layout()
        figname = "entrenamiento_%imodes_%ilayers.png" % (N_MODES, N_LAYERS)
        fig.canvas.draw()
        fig.savefig(output_dir + figname, dpi=200)

        # Grafico las series temporales de cada neurona y debajo la predicción de la red
        autoencoder_output = autoencoder(smoothed_sfa)

        fig, ax = plt.subplots(nrows=nseries, ncols=1, figsize=(16, 4 * smoothed_sfa.shape[1]))
        cmap = get_cmap("tab10")
        for col in range(nseries):
            ax[col].plot(smoothed_sfa[:, col], color=cmap(col % 10), ls="-", linewidth=.7, marker="", markersize=1,
                         label="Datos")
            ax[col].plot(autoencoder_output[:, col], color=cmap(col % 10), ls="--", linewidth=.7, marker="",
                         markersize=1, label="Predicción")
            ax[col].set_ylabel(f"Serie no. {col}")
            ax[col].set_xlabel(f"Tiempo [muestras]");
            ax[col].legend(loc="upper left")

            ax[col].minorticks_on()
            ax[col].tick_params(axis='x', which='major', bottom=True)
            ax[col].grid(axis="x", which="major", ls=":")
            ax[col].grid(axis="x", which="minor", ls=":")

        # ax[-1].set_xlim([1000, 1500]);

        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(output_dir + "predicción.png", dpi=200)

        # Grafico los modos del espacio latente
        encoder_output = encoder(smoothed_sfa)
        x = np.arange(encoder_output.shape[0])
        limits = [300, 540, 1200, 1270, 1750, 1800, nsamples]

        fig, ax = plt.subplots(figsize=(12, 5))

        for mode_idx in range(encoder_output.shape[1]):
            ax.plot(x, encoder_output[:, mode_idx], linewidth=.7, zorder=10, label=f'Modo {mode_idx + 1}')

        import matplotlib.transforms as mtransforms

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for lim_idx in range(len(limits) - 1):
            ax.axvspan(limits[lim_idx], limits[lim_idx + 1], linewidth=2, color=cmap(lim_idx % 10), alpha=0.1)
            ax.text(np.mean(limits[lim_idx:lim_idx + 2]), ax.get_ylim()[0],
                    f"Ventana {lim_idx}\n{limits[i]}:{limits[i + 1]}", horizontalalignment="center",
                    verticalalignment="bottom")

        ax.set_xlabel("Tiempo [muestras]")
        ax.set_ylabel("Activación [ua]")
        ax.minorticks_on()
        ax.tick_params(axis='x', which='major', bottom=True)
        plt.grid(axis="x", which="major", ls=":")
        plt.grid(axis="x", which="minor", ls=":")
        # plt.xlim([600, 1050])  # Esta ventana temporal parece interesante
        plt.legend();

        fig.tight_layout()
        figname = "modos_%imodes_%ilayers.png" % (N_MODES, N_LAYERS)
        fig.canvas.draw()
        fig.savefig(output_dir + figname, dpi=200)

        # Gráfico dinámico del embedding para los primeros 3 modos
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if N_MODES <= 3:
            modes = range(N_MODES)
        else:
            modes = [0, 1, 2]

        for i in range(len(limits) - 1):
            xslice = slice(limits[i], limits[i + 1])
            ax.plot(*tuple([encoder_output[xslice, i] for i in modes]),
                    linewidth=.7,
                    label=f"Ventana {i} ({limits[i]}:{limits[i + 1]})")
        ax.set_xlabel("Modo 1");
        ax.set_ylabel("Modo 2");
        ax.set_zlabel("Modo 3");
        ax.legend();
        ax.view_init(10, -45)  # Roto los ejes. Conviene ir variando esto para probar.
        # El orden es (theta, phi) de esféricas

        fig.tight_layout()
        figname = "atractor_%imodes_%ilayers.png" % (N_MODES, N_LAYERS)
        fig.canvas.draw()
        fig.savefig(output_dir + figname, dpi=200)

    # Gráfico dinámico del embedding para los primeros 3 modos
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if N_MODES <= 3:
        modes = range(N_MODES)
    else:
        modes = [0, 1, 2]

    for i in range(len(limits) - 1):
        xslice = slice(limits[i], limits[i + 1])
        ax.plot(*tuple([encoder_output[xslice, i] for i in modes]),
                linewidth=.7,
                label=f"Ventana {i} ({limits[i]}:{limits[i + 1]})")
    ax.set_xlabel("Modo 1");
    ax.set_ylabel("Modo 2");
    ax.set_zlabel("Modo 3");
    ax.legend();
    ax.view_init(10, -45)  # Roto los ejes. Conviene ir variando esto para probar.
    # El orden es (theta, phi) de esféricas

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(output_dir + "atractor.png", dpi=200)

    plt.close('all')
