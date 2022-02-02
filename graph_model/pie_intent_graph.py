
import time

from torch.autograd import Variable

import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from  sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from graph_model.utils import *
from graph_model.model import *
from graph_model.dataset import Dataset


from matplotlib import pyplot

class PIEIntent(object):
    """
    A convLSTM encoder decoder model for predicting pedestrian intention
    Attributes:
        _num_hidden_units: Number of LSTM hidden units
        _reg_value: the value of L2 regularizer for training
        _kernel_regularizer: Training regularizer set as L2
        _recurrent_regularizer: Training regularizer set as L2
        _activation: LSTM activations
        _lstm_dropout: input dropout
        _lstm_recurrent_dropout: recurrent dropout
        _convlstm_num_filters: number of filters in convLSTM
        _convlstm_kernel_size: kernel size in convLSTM
    Model attributes: set during training depending on the data
        _encoder_input_size: size of the encoder input
        _decoder_input_size: size of the encoder_output
    Methods:
        load_images_and_process: generates trajectories by sampling from pedestrian sequences
        get_data_slices: generate tracks for training/testing
        create_lstm_model: a helper function for creating conv LSTM unit
        pie_convlstm_encdec: generates intention prediction model
        train: trains the model
        test_chunk: tests the model (chunks the test cases for memory efficiency)
    """

    def __init__(self, n_stgcnn=4, n_txpcnn=0, seq_len=15, pred_seq_len=1, kernel_size=15):

        # Network parameters
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.obs_seq_len = seq_len
        self.kernel_size = kernel_size
        self.pred_seq_len = pred_seq_len

    def get_model_config(self):
        """
        Returns a dictionary containing model configuration.
        """

        config = dict()
        # Network parameters
        config['n_stgcnn'] = self.n_stgcnn
        config['n_txpcnn'] = self.n_txpcnn
        config['obs_seq_len'] = self.obs_seq_len
        config['pred_seq_len'] = self.pred_seq_len
        config['kernel_size'] = self.kernel_size

        print(config)
        return config

    def load_model_config(self, config):
        """
        Copy config information from the dictionary for testing
        """
        # Network parameters
        self._sequence_length = config['sequence_length']
        self.n_stgcnn = config['n_stgcnn']
        self.n_txpcnn = config['n_txpcnn']
        self.obs_seq_len = config['obs_seq_len']
        self.pred_seq_len = config['pred_seq_len']
        self.kernel_size = config['kernel_size']

    def get_path(self,
                 type_save='models',  # model or data
                 models_save_folder='',
                 model_name='',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder='./data/'):

        """
        A path generator method for saving model and config data. Creates directories
        as needed.
        :param type_save: Specifies whether data or model is saved.
        :param models_save_folder: model name (e.g. train function uses timestring "%d%b%Y-%Hh%Mm%Ss")
        :param model_name: model name (either trained convlstm_encdec model or vgg16)
        :param file_name: Actual file of the file (e.g. model.h5, history.h5, config.pkl)
        :param data_subset: train, test or val
        :param data_type: type of the data (e.g. features_context_pad_resize)
        :param save_root_folder: The root folder for saved data.
        :return: The full path for the save folder
        """
        assert (type_save in ['models', 'data'])
        if data_type != '':
            assert (any([d in data_type for d in ['images', 'features']]))
        root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'graph', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'graph', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def get_tracks(self, dataset, data_type, seq_length, overlap):
        """
        Generate tracks by sampling from pedestrian sequences
        :param dataset: raw data from the dataset
        :param data_type: types of data for encoder/decoder input
        :param seq_length: the length of the sequence
        :param overlap: defines the overlap between consecutive sequences (between 0 and 1)
        :return: a dictionary containing sampled tracks for each data modality
        """
        overlap_stride = seq_length if overlap == 0 else \
            int((1 - overlap) * seq_length)

        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        d_types = []
        for k in data_type.keys():
            d_types.extend(data_type[k])
        d = {}

        if 'bbox' in d_types:
            d['bbox'] = dataset['bbox']
        if 'intention_binary' in d_types:
            d['intention_binary'] = dataset['intention_binary']
        if 'intention_prob' in d_types:
            d['intention_prob'] = dataset['intention_prob']

        bboxes = dataset['bbox'].copy()
        images = dataset['image'].copy()
        ped_ids = dataset['ped_id'].copy()
        frame_ids = dataset['frame_id'].copy()

        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in \
                               range(0, len(track) \
                                     - seq_length + 1, overlap_stride)])
            d[k] = tracks

        pid = []
        for p in ped_ids:

            pid.extend([p[i:i + seq_length] for i in \
                        range(0, len(p) \
                              - seq_length + 1, overlap_stride)])

        ped_ids = pid

        frame = []
        for f in frame_ids:
            frame.extend([f[i:i + seq_length] for i in \
                        range(0, len(f) \
                              - seq_length + 1, overlap_stride)])
        frame_ids = frame

        im = []
        for img in images:
            im.extend([img[i:i + seq_length] for i in \
                       range(0, len(img) \
                             - seq_length + 1, overlap_stride)])
        images = im

        bb = []
        for bbox in bboxes:
            bb.extend([bbox[i:i + seq_length] for i in \
                       range(0, len(bbox) \
                             - seq_length + 1, overlap_stride)])

        bboxes = bb
        return d, images, bboxes, ped_ids, frame_ids

    def concat_data(self, data, data_type):
        """
        Concatenates different types of data specified by data_type.
        Creats dummy data if no data type is specified
        :param data_type: type of data (e.g. bbox)
        """
        if not data_type:
            return []
        # if more than one data type is specified, they are concatenated
        d = []

        for dt in data_type:
            d.append(np.array(data[dt]))

        if len(d) > 1:
            d = np.concatenate(d, axis=2)
        else:
            d = d[0]

        return d

    def get_train_val_data(self, data, data_type, seq_length, overlap, type, datasize = 500):
        """
        A helper function for data generation that combines different data types into a single
        representation.
        :param data: A dictionary of data types
        :param data_type: The data types defined for encoder and decoder
        :return: A unified data representation as a list.
        """

        tracks, images, bboxes, ped_ids, frame_ids = self.get_tracks(data, data_type, seq_length, overlap)

        # Generate observation data input to encoder
        encoder_input = self.concat_data(tracks, data_type['encoder_input_type'])
        decoder_input = self.concat_data(tracks, data_type['decoder_input_type'])
        output = self.concat_data(tracks, data_type['output_type'])

        if len(decoder_input) == 0:
            decoder_input = np.zeros(shape=np.array(bboxes).shape)

        images_s = images
        bboxes_s = bboxes
        ped_ids_s = ped_ids
        frame_ids_s = frame_ids
        decoder_input_s = decoder_input
        output_s = output

        return {'images': images_s,
                'bboxes': bboxes_s,
                'ped_ids': ped_ids_s,
                'frame_ids': frame_ids_s,
                'encoder_input': encoder_input,
                'decoder_input': np.asarray(decoder_input_s),
                'output': output_s}

    def get_model(self, max_nodes):

        train_model = social_stgcnn(max_nodes=max_nodes).cuda()

        return train_model

    def train(self,
              data_train,
              data_val,
              batch_size=128,
              epochs=400,
              optimizer_type='sgd',
              optimizer_params={'lr': 0.001, 'clipvalue': 0.0, 'decay': 0},
              loss=['binary_crossentropy'],
              metrics=['acc'],
              data_opts='',
              datasize = 500,
              first_time=False,
              path='',
              node_info =''):

        """
        Training method for the model
        :param data_train: training data
        :param data_val: validation data
        :param batch_size: batch size for training
        :param epochs: number of epochs for training
        :param optimizer_params: learning rate and clipvalue for gradient clipping
        :param loss: type of loss function
        :param metrics: metrics to monitor
        :param data_opts: data generation parameters
        """
        data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                     'decoder_input_type': data_opts['decoder_input_type'],
                     'output_type': data_opts['output_type']}

        train_config = {'batch_size': batch_size,
                        'epoch': epochs,
                        'optimizer_type': optimizer_type,
                        'optimizer_params': optimizer_params,
                        'loss': loss,
                        'metrics': metrics,
                        'learning_scheduler_mode': 'plateau',
                        'lambda_l2': 0.05,
                        'torch.seed': torch.initial_seed(),
                        'learning_scheduler_params': {'exp_decay_param': 0.3,
                                                      'step_drop_rate': 0.5,
                                                      'epochs_drop_rate': 20.0,
                                                      'plateau_patience': 5,
                                                      'min_lr': 0.0000001,
                                                      'monitor_value': 'val_loss'},
                        'model': 'social-stgcn',
                        'data_type': data_type,
                        'overlap': data_opts['seq_overlap_rate'],
                        'dataset': 'pie'}

        self._model_type = train_config['model']
        seq_length = data_opts['max_size_observe']
        
        train_d = self.get_train_val_data(data_train, data_type, seq_length, 0.5, 'train', datasize)
        val_d = self.get_train_val_data(data_val, data_type, seq_length, 0, 'val')


        self._encoder_seq_length = train_d['decoder_input'].shape[1]
        self._decoder_seq_length = train_d['decoder_input'].shape[1]

        self._sequence_length = self._encoder_seq_length

        train_dataset = Dataset(data_train, train_d, data_opts, 'train',  first_time=first_time, path=path, node_info=node_info)
        val_dataset = Dataset(data_val, val_d, data_opts, 'val', first_time=first_time, path=path, node_info=node_info)


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=128, shuffle=True, num_workers=4,
                                                   pin_memory=False)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=128, shuffle=False, num_workers=4,
                                                pin_memory=False)

        # automatically generate model name as a time string
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        model_path, _ = self.get_path(type_save='models',
                                      model_name=train_config['model'],
                                      models_save_folder=model_folder_name,
                                      file_name='model.pth',
                                      save_root_folder='data')
        config_path, _ = self.get_path(type_save='models',
                                       model_name=train_config['model'],
                                       models_save_folder=model_folder_name,
                                       file_name='configs',
                                       save_root_folder='data')

        max_path, _ = self.get_path(type_save='models',
                                       model_name=train_config['model'],
                                       models_save_folder=model_folder_name,
                                       file_name='max',
                                       save_root_folder='data')

        # Save config and training param files
        with open(config_path + '.pkl', 'wb') as fid:
            pickle.dump([self.get_model_config(),
                         train_config, data_opts],
                        fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote configs to {}'.format(config_path))

        # Save config and training param files
        with open(config_path + '.txt', 'wt') as fid:
            fid.write("####### Model config #######\n")
            fid.write(str(self.get_model_config()))
            fid.write("\n####### Training config #######\n")
            fid.write(str(train_config))
            fid.write("\n####### Data options #######\n")
            fid.write(str(data_opts))

        train_model = social_stgcnn(max_nodes=train_dataset.max_nodes).cuda()


        optimizer = torch.optim.SGD(train_model.parameters(),
                                    lr=optimizer_params['lr'],
                                    momentum=0.9)

        loss_fn = nn.BCEWithLogitsLoss()

        #################################################################################################

        epoch_losses_train = []
        epoch_accuracy_train = []
        best_accuracy = 0
        epoch_losses_val = []
        epoch_accuracy_val = []

        max_TN = 0.5
        max_acc = 0
        max_epochs = 0

        for epoch in range(epochs):
            accuracy = 0
            count = 0
            print("###########################")
            print("######## NEW EPOCH ########")
            print("###########################")
            # print("epoch: %d/%d" % (epoch + 1, epochs), "lr: ", scheduler.get_lr()[0])
            print("epoch: %d/%d" % (epoch + 1, epochs))
            ###########################################################################
            # train:
            ###########################################################################
            train_model.train()  # (set in training mode, this affects BatchNorm and dropout)
            batch_losses = []
            y_true = []
            y_pred = []
            y_s_pred = []
            count_cross_p, count_cross_Tp = 0, 0
            count_cross_n, count_cross_Tn = 0, 0
            newvari = []

            for step, (graph, adj_matrix, location, label, node_label,class_label) in enumerate(train_loader):

                G = Variable(graph.type(torch.FloatTensor)).cuda()
                Adj = Variable(adj_matrix.type(torch.FloatTensor)).cuda()
                node_label = Variable(node_label.type(torch.FloatTensor)).cuda()
                class_label = Variable(class_label.type(torch.FloatTensor)).cuda()
                Loc = Variable(location.type(torch.FloatTensor)).cuda()

                label = Variable(label.type(torch.float)).cuda()

                outputs = train_model(G, Adj, Loc, node_label,class_label)


                l2_enc = torch.cat([x.view(-1) for x in train_model.st_gcn_networks.parameters()])
                l2_enc_loc = torch.cat([x.view(-1) for x in train_model.st_gcn_networks_loc.parameters()])
                l2_dec = torch.cat([x.view(-1) for x in train_model.dec.parameters()])

                loss = loss_fn(outputs, label)
                batch_losses.append(loss.data.cpu().numpy())

                loss += 0.01 * torch.norm(l2_dec, p=1) + \
                        0.001 * torch.norm(l2_enc_loc, p=2) + \
                        0.05 * torch.norm(l2_enc, p=2)

                


                optimizer.zero_grad()  # (reset gradients)
                loss.backward()  # (compute gradients)
                optimizer.step()  # (perform optimization step)

                y_true.append(np.asarray(label.data.to('cpu')))
                y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))


            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            print('Correct predictions of non-crossing', count_cross_n)
            print('Total non-crossing', count_cross_Tn)
            print('Correct predictions of crossing', count_cross_p)
            print('Total crossing', count_cross_Tp)


            TN = confusion_matrix(y_true, y_pred)[0, 0]
            FP = confusion_matrix(y_true, y_pred)[0, 1]
            FN = confusion_matrix(y_true, y_pred)[1, 0]
            TP = confusion_matrix(y_true, y_pred)[1, 1]
            accuracy = accuracy_score(y_true, y_pred)
            epoch_loss = np.mean(batch_losses)
            epoch_losses_train.append(epoch_loss)
            epoch_accuracy_train.append(accuracy)
            print("train loss: %g" % epoch_loss)
            print("Accuracy:  %g" % accuracy)
            print('CONFUSION MATRIX:')
            print("TP: %g" % TP, "FP: %g" % FP)
            print("FN: %g" % FN, "TN: %g" % TN)
            print("####")

   
            # ############################################################################
            # # val:
            # ############################################################################
            train_model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
            batch_losses = []

            y_true = []
            y_pred = []
            y_s_pred = []
            val_loss = 0
            count = 0
            count_cross_p, count_cross_Tp = 0, 0
            count_cross_n, count_cross_Tn = 0, 0
            for step, (graph, adj_matrix, location, label, node_label,class_label) in enumerate(val_loader):
                with torch.no_grad():

                    G = Variable(graph.type(torch.FloatTensor)).cuda()
                    Adj = Variable(adj_matrix.type(torch.FloatTensor)).cuda()
                    node_label = Variable(node_label.type(torch.FloatTensor)).cuda()
                    class_label = Variable(class_label.type(torch.FloatTensor)).cuda()
                    Loc = Variable(location.type(torch.FloatTensor)).cuda()

                    label = Variable(label.type(torch.float)).cuda()

                    outputs = train_model(G, Adj, Loc, node_label,class_label)

                    l2_enc = torch.cat([x.view(-1) for x in train_model.st_gcn_networks.parameters()])
                    l2_enc_loc = torch.cat([x.view(-1) for x in train_model.st_gcn_networks_loc.parameters()])
                    l2_dec = torch.cat([x.view(-1) for x in train_model.dec.parameters()])

                    loss = loss_fn(outputs, label)
                    batch_losses.append(loss.data.cpu().numpy())

                    loss += 0.01 * torch.norm(l2_dec, p=1) + \
                            0.001 * torch.norm(l2_enc_loc, p=2) + \
                            0.05 * torch.norm(l2_enc, p=2)

                    y_true.append(np.asarray(label.data.to('cpu')))
                    y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))
                    y_s_pred.append(torch.sigmoid(outputs).data.to('cpu'))


            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_s_pred = np.concatenate(y_s_pred, axis=0)
            print('Correct predictions of non-crossing', count_cross_n)
            print('Total non-crossing', count_cross_Tn)
            print('Correct predictions of crossing', count_cross_p)
            print('Total crossing', count_cross_Tp)

            TN = confusion_matrix(y_true, y_pred)[0, 0]
            FP = confusion_matrix(y_true, y_pred)[0, 1]
            FN = confusion_matrix(y_true, y_pred)[1, 0]
            TP = confusion_matrix(y_true, y_pred)[1, 1]
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_s_pred)
            avg_p = average_precision_score(y_true, y_s_pred)

            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            epoch_accuracy_val.append(accuracy)

            print("val loss: %g" % epoch_loss)
            print("Accuracy:  %g" % accuracy)
            print('CONFUSION MATRIX:')
            print("TP: %g" % TP, "FP: %g" % FP)
            print("FN: %g" % FN, "TN: %g" % TN)

            print('Precision:', precision)
            print('Recall:', recall)
            print('F1 score:', f1)
            print('ROC AUC:', auc)
            print('Average Precision:', avg_p)

            plt.figure(1)
            plt.plot(epoch_losses_train, "r^")
            plt.plot(epoch_losses_train, "r", label='train')
            plt.plot(epoch_losses_val, "k^")
            plt.plot(epoch_losses_val, "k", label='val')
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.title("loss per epoch")
            plt.legend()
            plt.grid()
            plt.savefig("%s/epoch_losses.png" % model_path.split("model.pth")[0])
            plt.close(1)

            plt.figure(2)
            plt.plot(epoch_accuracy_train, "r^")
            plt.plot(epoch_accuracy_train, "r", label='train')
            plt.plot(epoch_accuracy_val, "k^")
            plt.plot(epoch_accuracy_val, "k", label='val')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.title("accuracy per epoch")
            plt.legend()
            plt.grid()
            plt.savefig("%s/epoch_accuracy.png" % model_path.split("model.pth")[0])
            plt.close(2)

            save = False
            if auc >= max_acc:
                if TN/(TN+FP) > 0.5:
                    max_acc = auc
                    save = True
            print(save)
            if save is True:
                model_saving_path = model_path.split("model.pth")[0] + "/model_" + "epoch_best.pth"
                torch.save(train_model.state_dict(), model_saving_path)

        return model_path.split("model.pth")[0]
    #############################################################################################
    # Testing code
    #############################################################################################

    # split test data into chunks
    def test_chunk(self,
                   data_test,
                   data_opts='',
                   model_path='',
                   first_time = False,
                   path = '',
                   node_info=''):

        with open(os.path.join(model_path, 'configs.pkl'), 'rb') as fid:
            try:
                configs = pickle.load(fid)
            except:
                configs = pickle.load(fid, encoding='bytes')

        train_params = configs[1]
        seq_length = configs[2]['max_size_observe']
        overlap = 0
        tracks, images, bboxes, ped_ids, frame_ids = self.get_tracks(data_test,
                                                          train_params['data_type'],
                                                          seq_length,
                                                          overlap)

        # Generate observation data input to encoder
        decoder_input = self.concat_data(tracks, train_params['data_type']['decoder_input_type'])

        output = self.concat_data(tracks, train_params['data_type']['output_type'])
        if len(decoder_input) == 0:
           decoder_input = np.zeros(shape=np.array(bboxes).shape)

        test_d = {'images': images,
                'bboxes': bboxes,
                'ped_ids': ped_ids,
                'frame_ids': frame_ids,
                'decoder_input': decoder_input,
                'output': output}

        test_dataset = Dataset(data_test, test_d, data_opts, 'test', first_time= first_time, path=path, node_info=node_info)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=128, shuffle=False,
                                                  pin_memory=False,
                                                  num_workers=4)

        test_model = self.get_model(test_dataset.max_nodes)

        test_model.load_state_dict(torch.load(os.path.join(model_path, 'model_epoch_best.pth')))

        test_model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)

        count = 0
        some_count = 0
        y_true = []
        y_pred = []
        y_s_pred = []
        count_cross_p = 0
        count_cross_n = 0
        with open(model_path + '/misclassification.txt', 'wt') as fid:
            fid.write("####### Misclssification #######\n")

            for step, (graph, adj_matrix, location, label, node_label,class_label) in enumerate(test_loader):
                with torch.no_grad():
                    if count % 10 == 0:
                        print(count)
                    count = count + 1

                    G = Variable(graph.type(torch.FloatTensor)).cuda()
                    # G = []
                    Adj = Variable(adj_matrix.type(torch.FloatTensor)).cuda()
                    node_label = Variable(node_label.type(torch.FloatTensor)).cuda()
                    class_label = Variable(class_label.type(torch.FloatTensor)).cuda()
                    Loc = Variable(location.type(torch.FloatTensor)).cuda()
                    label = Variable(label.type(torch.float)).cuda()
                    
                    outputs = test_model(G, Adj, Loc, node_label,class_label)

                    y_true.append(np.asarray(label.data.to('cpu')))
                    y_pred.append(np.round(torch.sigmoid(outputs).data.to('cpu')))
                    y_s_pred.append(torch.sigmoid(outputs).data.to('cpu'))


        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_s_pred = np.concatenate(y_s_pred, axis=0)
        print(count_cross_p)
        print(count_cross_n)
        count_totalp = 0
        score_p = 0
        count_p = 0
        count_totaln = 0
        score_n = 0
        count_n = 0
        for test_gt, test_pt in zip(y_true, y_s_pred):
            if int(test_gt) == 1:
                count_totalp += 1
                score_p += test_pt
                if np.round(test_pt) == 1:
                    count_p += 1
            else:
                count_totaln += 1
                score_n += test_pt
                if np.round(test_pt) == 0:
                    count_n += 1

        delta = (count_p / count_totalp) - (count_n / count_totaln)
        delta_s = (score_p / count_totalp) - (score_n / count_totaln)
        auc = roc_auc_score(y_true, y_s_pred)
        avg_p = average_precision_score(y_true, y_s_pred)

        TN = confusion_matrix(y_true, y_pred)[0, 0]
        FP = confusion_matrix(y_true, y_pred)[0, 1]
        FN = confusion_matrix(y_true, y_pred)[1, 0]
        TP = confusion_matrix(y_true, y_pred)[1, 1]

        print('CONFUSION MATRIX:')
        print("TP: %g" % TP, "FP: %g" % FP)
        print("FN: %g" % FN, "TN: %g" % TN)
        print('ROC AUC:', auc)
        print('Average Precision:', avg_p)
        print('Delta predictions:', delta)
        print('Delta score:', delta_s)

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        return acc, f1, precision, recall