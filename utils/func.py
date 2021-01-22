import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, classification_report, recall_score, precision_score
from sklearn.neighbors import KernelDensity
from tensorflow.keras import layers, models
import tensorflow as tf

def fillNan(row):
    tenure = row['tenure']
    if tenure == 0:
        monthly_charges = row['MonthlyCharges']
        row['TotalCharges'] = monthly_charges
        return row
    return row


def visualize(df, feature, target):
    
    """ 
    utility function to help visualization
    """
    
    _df = df.groupby([feature.name, target.name]).size().reset_index().pivot(columns = target.name, index = feature.name, values = 0)
    _df.plot(kind = 'bar', stacked = True)
    print(feature.value_counts(),'\n')
    indexes = _df.index
    scores = _df.to_numpy()[:,1] / _df.to_numpy().sum(axis = 1)
    for i, index in enumerate(indexes):
        print(f' {index} : churned customers : {scores[i]:.3f}')


def plot_density_estimator_1D(df, serie, figsize = (9, 3)):
    
    """
    function to plot KDE
    """
    
    datas = []
    for i,v in {'Yes': 'Churn',  'No': 'Not churn'}.items():
        data = pd.DataFrame(serie.loc[df['Churn'] == i])
        datas.append(data)
        
        q1 = data.quantile(0.25)[0]
        q3 = data.quantile(0.75)[0]
        sigma = data.std()[0]
        m = len(data)
        h = 0.9 * min(sigma, (q3 - q1) / 1.34) * m **(-0.2)
        kde = KernelDensity(kernel = 'gaussian', bandwidth = h)
        kde.fit(data)
        
        vmax = data.max()[0]
        vmin = data.min()[0]
        xr = np.linspace(vmin, vmax, 100)
        xvals = xr.reshape((-1, 1))
        
        dvals = np.exp(kde.score_samples(xvals))
        
        plt.plot(xvals, dvals, label = v) 
        plt.xlabel(serie.name)
        plt.ylabel('density')
        plt.legend()
        plt.tight_layout()
        
    return datas

def plot_boxplots(df):

	fig, ax = plt.subplots(1, 3)
	ax[0].boxplot(df['tenure'], widths = 0.4, patch_artist = True)
	ax[0].set_title('tenure')

	ax[1].boxplot(df['MonthlyCharges'], widths = 0.4, patch_artist = True)
	ax[1].set_title('MonthlyCharges')

	ax[2].boxplot(df['TotalCharges'], widths = 0.4, patch_artist = True)
	ax[2].set_title('TotalCharges')

	fig.subplots_adjust(left = 0.5, right = 2, bottom = 0.1,
	                    top = 1, hspace = 20, wspace = 1)

	plt.show()


def grid_search_selector(tr, vl, models, params):
    
    """
    search the best parameters for the given models
    """

    best_params = {}
    best_scores = {}
    
    X_tr = tr[0]
    Y_tr = tr[1]
    
    X_vl = vl[0]
    Y_vl = vl[1]
    
    for n, (name, model) in enumerate(models.items()):
        
        clf = GridSearchCV(estimator = model, param_grid = params[n], cv = 5).fit(X_tr, Y_tr)
        best_params[name] = clf.best_params_
        best_scores[name] = clf.score(X_vl, Y_vl)
        print(f'{str(name)} -- {best_scores[name]}')
    
    return best_params, best_scores



def model_evaluation(model, params, tr, ts):
    
    """
    evaluate the given model
    """

    X_tr = tr[0]
    Y_tr = tr[1]
    
    X_ts = ts[0]
    Y_ts = ts[1]
    
    model = model.set_params(**params)
    model.fit(X_tr, Y_tr)
    
    y_pred_train = model.predict(X_tr)
    y_pred = model.predict(X_ts)
    
    acc_score_train = accuracy_score(Y_tr, y_pred_train)
    acc_score = accuracy_score(Y_ts, y_pred)
    
    print('report on the test set : ')
    print(classification_report(Y_ts, y_pred, target_names = ['Non-churned', 'Churned']))
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    plot_roc_curve(model, X_ts, Y_ts, ax = axes[0])
    sns.lineplot([0,1], [0,1], ax = axes[0])
    plot_confusion_matrix(model, X_ts, Y_ts, display_labels= ['Non-churned', 'Churned'], cmap = 'GnBu', ax = axes[1])
    score = confusion_matrix(Y_ts, y_pred)
    score = score.diagonal()/score.sum(axis = 1)
    print(f' test set accuracy non-churned customers : {score[0]:.4f}')
    print(f' test set accuracy churned customers : {score[1]:.4f}')
    plt.tight_layout()

    return model

def build_classifier(input_shape, hidden = []):

    model_in = layers.Input( shape = input_shape, dtype = 'float32')
    x = model_in

    for h in hidden:
        x = layers.Dense(h, activation = 'relu')(x)
    model_out = layers.Dense(1, activation = 'sigmoid')(x)
    model = models.Model(model_in, model_out)
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = 'accuracy')

    return model

def compute_metric(x, y, model):
    
    y_pred = model.predict(x)
    y_pred = np.round(y_pred)
    y_pred = np.int32(y_pred) 
    score = confusion_matrix(y, y_pred)
    score = score.diagonal()/score.sum(axis = 1)
    return score, np.round(accuracy_score(y, y_pred), 2)

def plot_training_history(history, figsize = (15,6), autoclose = True):
    
    f_loss = np.min(history.history['loss'])
    f_vloss = np.min(history.history['val_loss'])
    
    print(f'final loss : {f_loss:.4f} (training), {f_vloss:.4f} (validation)')
    
    if autoclose:
        plt.close('all')
    fig = plt.figure(figsize = figsize)
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    ax1 = plt.subplot(1, 2, 1)
    
    ax1.plot(history.history['loss'], label = 'tr. loss')
    ax1.plot(history.history['val_loss'], label = 'val. loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.legend()
    
    ax2 = plt.subplot(1, 2, 2)
    
    ax2.plot(history.history['accuracy'], label = 'tr. accuracy')
    ax2.plot(history.history['val_accuracy'], label = 'val. accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    
    plt.legend()
    plt.tight_layout()


def build_autoencoder(input_shape, hidden = [15]):

	hidden = sorted(hidden, reverse = True)

	encoder_in = layers.Input(shape = input_shape, dtype = 'float32')

	encoder_hidden = layers.Dense(hidden[0], activation = 'relu')(encoder_in)
	for h in hidden[1:-1]:
		encoder_hidden = layers.Dense(h, activation = 'relu')(encoder_hidden)
	encoder_out = layers.Dense(hidden[-1], activation = 'relu')(encoder_hidden)

	encoder = models.Model(encoder_in, encoder_out, name = 'encoder')

	decoder_input = layers.Input(shape = (hidden[-1],), dtype = 'float32')

	if len(hidden) > 1:
		decoder_hidden = layers.Dense(hidden[-2], activation = 'relu')(decoder_input)
		if len(hidden) > 2:
			for h in hidden[-3::-1]:
				decoder_hidden = layers.Dense(h, activation = 'relu')(decoder_hidden)

		decoder_output = layers.Dense(input_shape[0], activation = 'linear')(decoder_hidden)
	else:
		decoder_output = layers.Dense(input_shape[0], activation = 'linear')(decoder_input)

	decoder = models.Model(decoder_input, decoder_output, name = 'decoder')

	outputs = decoder(encoder(encoder_in))

	encoder_decoder = models.Model(encoder_in, outputs, name = 'encoder_decoder')
	encoder_decoder.compile(optimizer = 'RMSProp', loss = 'mse', metrics = ['accuracy'])
	
	return encoder_decoder, encoder, decoder


def plot_reconstruction_errors(x, y, model = None, models = None):

	if models == None:
		preds = model.predict(x)
	else:
		encoder = models[0]
		decoder = models[1]

		x_enc = encoder.predict(x)
		preds = decoder.predict(x_enc)

	mse = np.mean(np.power(x - preds, 2), axis = 1)

	error_df = pd.DataFrame({'recon_errors': mse, 'churn': y})

	plt.figure(figsize = (10, 6))
	sns.kdeplot(error_df.recon_errors[error_df.churn == 0], label = 'Not churn', shade = True, clip = (0, 10))
	sns.kdeplot(error_df.recon_errors[error_df.churn == 1], label = 'Churn', shade = True, clip = (0, 10))
	plt.xlabel('reconstruction error')
	plt.legend()

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(preds, labels):
    print("Accuracy = {}".format(accuracy_score(labels, preds)))
    print("Precision = {}".format(precision_score(labels, preds)))
    print("Recall = {}".format(recall_score(labels, preds)))


def plot_signal(signal, targets = None, figsize = (15, 6), autoclose = True, s = 0.5):
    if autoclose:
        plt.close('all')
    plt.figure(figsize = figsize)
    plt.bar(signal.index, signal, label = 'signal')
    if targets is not None:
        nonzero = signal.index[targets != 0] # not churn customers
        smin, smax = np.min(signal),  np.max(signal)
        lvl = smin - 0.05 * (smax-smin)
        plt.scatter(nonzero, np.ones(len(nonzero)) * lvl, s = s, color = 'tab:orange')
    plt.tight_layout()



def plot_dataframe(data, targets = None, figsize = (15, 6), autoclose = True, s = 4):
    if autoclose: plt.close('all')
    plt.figure(figsize = figsize)
    # the values lower than vmin or largin than vmax will be displyed with the same color
    plt.imshow(data.T, aspect = 'auto', cmap = 'RdBu', vmin = -1.96, vmax = 1.96)
    if targets is not None:
        nonzero = data.index[targets != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        plt.scatter(nonzero, lvl*np.ones(len(nonzero)), s = s, color = 'tab:orange')
        plt.scatter(targets.index, np.ones(len(targets)) * lvl, s = s, color = plt.get_cmap('tab10')(targets))
    plt.tight_layout()

def plot_latent_space(x, y, encoder, dims = (0,1)):

	preds = encoder.predict(x)
	plt.figure(figsize = (15, 6))
	plt.scatter(preds[:, dims[0]], preds[:, dims[1]], c = y, alpha = 0.6)
	plt.title('Latent space')
	plt.show()

	return preds

def plot_bars(data, figsize = (15,6), autoclose = True, tick_gap = 1):
    if autoclose: plt.close('all')
    plt.figure(figsize = figsize)
    x = 0.5 + np.arange(len(data))
    plt.bar(x, data, width = 0.7)
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation = 45)
    plt.tight_layout()


def knn_visualization(x, y, model, dims = (0, 1), h = 0.2):

	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

	x_min, x_max = x[:, dims[0]].min() - 1, x[:, dims[0]].max() + 1
	y_min, y_max = x[:, dims[1]].min() - 1, x[:, dims[1]].max() + 1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.figure(figsize = (15, 6))
	plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

	plt.scatter(x[:, 0], x[:, 1], c = y, cmap = cmap_bold, edgecolors = 'k', s = 20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())



# build variational autoencoder
def build_vae(batch_size):
    
    def sample_z(args):
        # we cannot use backpropagation if we use random sampling
        # we can leverage this by using the 'reparameterization trick' which suggest that we randomly sample eps from a 
        # unit gaussian, and then shift the randomly sampled eps by the latent distribution's mean and scale it by the
        # latent distribution's variance
        z_mean, z_log_sigma = args
        #eps = tf.random.normal(shape = (batch_size, 2))
        #return mean + tf.math.exp(log_sigma / 2) * eps # z = mean + sigma * eps
        eps = K.random_normal(shape = (batch_size, 2), mean = 0., stddev = 1.)
        return z_mean + K.exp(z_log_sigma / 2) * eps
    
    def vae_loss(y_true, y_pred):
        # caluclate the loss = reconstruction loss + KL loss
        #recon = tf.math.reduce_sum(tf.keras.losses.binary_crossentropy(y_pred, y_true), axis = 1)
        #kl = 0.5 * tf.math.reduce_sum(tf.math.exp(log_sigma) + tf.math.square(mean) -1 - log_sigma, axis = 1)
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis = 1)
        kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) -1 - z_log_sigma, axis = 1)
        return recon + kl
    
    input_shape = (X_tr.shape[1],)
    encoder_in = layers.Input(shape = input_shape, dtype = 'float32', name = 'encoder_in')
    
    encoder_hidden = layers.Dense(512, activation = 'relu', name = 'encoder_hidden')(encoder_in)
    
    # Rather than directly outputting values for the lattent state as we would in a 
    # standard autoencoder, the encoder model of a VAE will output parameters
    # describing a distribution for each dimension in the latent space. Since we are
    # assuming that our prior follows a normal distribution, we will output 2 vectors describing the
    # mean and the variance of the latent space distributions.
    z_mean = layers.Dense(2, name = 'z_mean')(encoder_hidden)
    z_log_sigma = layers.Dense(2, name = 'z_log_sigma')(encoder_hidden)
    
    encoder = keras.Model(encoder_in, z_mean, name = 'encoder_model') # the encoder model
    
    # sample z ~ Q(z|X)
    z = layers.Lambda(sample_z, name = 'sampleZ')([z_mean, z_log_sigma])
    
    decoder_hidden = layers.Dense(512, activation = 'relu', name = 'decoder_hidden')(z)
    decoder_out = layers.Dense(input_shape[0], activation = 'sigmoid', name = 'decoder_output') (decoder_hidden)
    
    vae = keras.Model(encoder_in, decoder_out, name = 'vae_model') # the whole model (encoder + decoder)
    
    # generator model, generate new data given latent variable z
    d_in = layers.Input(shape = (2,), name = 'decoder_in')
    d_h = layers.Dense(512, activation = 'relu')(d_in)
    d_out = layers.Dense(input_shape[0], activation = 'sigmoid')(d_h)
    decoder = keras.Model(d_in, d_out, name = 'decoder_model')
    
    vae.compile(optimizer = 'adam', loss = vae_loss, metrics =['accuracy'])
    return vae, encoder, decoder