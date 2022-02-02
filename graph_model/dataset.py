import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable


import math


from graph_model.utils import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, track, data_opts, data_type,  first_time =False, path = '', node_info = ''):

        self.track = track
        self.dataset = dataset
        self.data_opts = data_opts
        self.data_type = data_type

        self.img_sequences = self.track['images']
        self.ped_ids = self.track['ped_ids']
        self.bbox_sequences = self.track['bboxes']
        self.decoder_input = self.track['decoder_input']
        self.unique_frames = self.dataset['unique_frame']
        self.unique_ped = self.dataset['unique_ped']
        self.unique_bbox = self.dataset['unique_bbox']
        self.unique_image = self.dataset['unique_image']

        self.node_info = node_info

        self.structure = 'star'  # 'fully_connected'

        self.max_nodes = self.node_info['pedestrian'] + self.node_info['vehicle'] +\
                         self.node_info['traffic_light'] + self.node_info['crosswalk'] +\
                         self.node_info['transit_station'] + self.node_info['sign'] +\
                         self.node_info['ego_vehicle']

        self.path = path  # Folder where the images are saved

        self.num_examples = len(self.track['ped_ids'])

        print(self.num_examples)
        print('Loading %s topology...' % str(self.structure))
        save_path = self.get_path(type_save='data',
                                  data_type='features' + '_' + self.data_opts[
                                      'crop_type'] + '_' + self.data_opts[
                                                'crop_mode'],  # images
                                  model_name='vgg16_bn',
                                  data_subset=self.data_type, ind=1)

        if first_time:
            vgg16 = models.vgg16_bn(pretrained=True).cuda()
            self.context_model = nn.Sequential(*list(vgg16.children())[:-1])

            try:
                self.convnet = self.context_model
            except:
                raise Exception("No context model is defined")

        self.feature_save_folder = './data/nodes_and_features/' + str(self.data_type)
        self.seq_len = len(self.track['images'][0])
        
        # False for stop creating pickle file for nodes, graph and adjacency matrix

        count_positive_samples = 0
        count_ped = []
        count_veh=[]
        count_sign=[]
        count_traflig=[]
        count_transit=[]
        count_cross = []

        count_object_samples = {'pedestrian': 0,  # default should be one
                     'vehicle': 0,
                     'traffic_light': 0,
                     'transit_station': 0,
                     'sign': 0,
                     'crosswalk': 0,
                     'ego_vehicle': 0}

        pose_donotexit = 0
        if True:
            i = -1
            for img_sequences, bbox_sequences, ped_ids in zip(self.track['images'],
                                                              self.track['bboxes'],
                                                              self.track['ped_ids']):

                i += 1
                seq_len = len(img_sequences)
                nodes = []
                max_nodes = 0
                node_features = []
                img_centre_seq = []
                bbox_location_seq = []
                primary_pedestrian_pose = []
                if int(self.track['output'][i][0]) == 1:
                    count_positive_samples += 1
                for imp, b, p in zip(img_sequences, bbox_sequences, ped_ids):

                    update_progress(i / self.num_examples)
                    imp = imp.replace(os.sep, '/')
                    set_id = imp.split('/')[-3]
                    vid_id = imp.split('/')[-2]
                    img_name = imp.split('/')[-1].split('.')[0]

                    key = str(set_id + vid_id)
                    frames = self.unique_frames[key].tolist()

                    ped = self.unique_ped[key]
                    box = self.unique_bbox[key]
                    image = self.unique_image[key]
                    index = frames.index(int(img_name))

                    if max_nodes < len(ped[index]):
                        max_nodes = len(ped[index])

                    img_features_unsorted = {}
                    img_centre_unsorted = {}
                    bbox_location_unsorted = {}
                    # Make sure that when data is loaded the pedestrian is the first key
                    for object_keys in ped[0]:
                        img_features_unsorted[object_keys] = []
                        img_centre_unsorted[object_keys] = []
                        bbox_location_unsorted[object_keys] = []

                        if not ped[index][object_keys]:
                            continue

                        if object_keys == 'pedestrian':
                            for idx, (n, bb, im) in enumerate(
                                    zip(ped[index][object_keys], box[index][object_keys], image[index][object_keys])):

                                if p == n:
                                    img_save_folder = os.path.join(save_path, set_id, vid_id)
                                    im = im.replace(os.sep, '/')
                                    im_name = im.split('/')[-1].split('.')[0]
                                    img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')

                                    if not os.path.exists(img_save_path):

                                        img_folder = os.path.join(self.path, set_id, vid_id)
                                        img_path = os.path.join(img_folder, im_name + '.png')
                                        img_data = load_img(img_path)
                                        bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
                                        bbox = squarify(bbox, 1, img_data.size[0])
                                        bbox = list(map(int, bbox[0:4]))
                                        cropped_image = img_data.crop(bbox)
                                        img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                                        image_array = img_to_array(img_data).reshape(-1, 224, 224)
                                        image_array = Variable(
                                            torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                                        image_features = self.convnet(image_array)
                                        image_features = image_features.data.to('cpu').numpy()
                                        if not os.path.exists(img_save_folder):
                                            os.makedirs(img_save_folder)
                                        with open(img_save_path, 'wb') as fid:
                                            pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)

                                    img_features_unsorted[object_keys].append([img_save_path])
                                    img_centre_unsorted[object_keys].append(self.get_center(bb))
                                    bbox_location_unsorted[object_keys].append(bb)
                                    key = os.path.join( im_name + '_' + n[0])

                        for idx, (n, bb, im) in enumerate(
                                zip(ped[index][object_keys], box[index][object_keys], image[index][object_keys])):

                            if p != n:

                                img_save_folder = os.path.join(save_path, set_id, vid_id)
                                im = im.replace(os.sep, '/')
                                im_name = im.split('/')[-1].split('.')[0]

                                img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')
                                if not os.path.exists(img_save_path):

                                    img_folder = os.path.join(self.path, set_id, vid_id)
                                    img_path = os.path.join(img_folder, im_name + '.png')
                                    img_data = load_img(img_path)
                                    if object_keys == 'ego_vehicle':
                                        bbox = jitter_bbox(img_path, [bb], 'same', 2)[0]
                                    else:
                                        bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
                                    bbox = squarify(bbox, 1, img_data.size[0])
                                    bbox = list(map(int, bbox[0:4]))
                                    cropped_image = img_data.crop(bbox)
                                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                                    image_array = img_to_array(img_data).reshape(-1, 224, 224)
                                    image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                                    image_features = self.convnet(image_array)
                                    image_features = image_features.data.to('cpu').numpy()
                                    if not os.path.exists(img_save_folder):
                                        os.makedirs(img_save_folder)
                                    with open(img_save_path, 'wb') as fid:
                                        pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)

                                img_features_unsorted[object_keys].append([img_save_path])
                                img_centre_unsorted[object_keys].append(self.get_center(bb))
                                bbox_location_unsorted[object_keys].append(bb)

                    # features, and their centre location in each frame

                    img_features = {}
                    bbox_location = {}
                    img_centre = {}

                    for object_keys in ped[0]:
                        img_features[object_keys] = []
                        bbox_location[object_keys] = []
                        img_centre[object_keys] = []

                        if not img_centre_unsorted[object_keys]:
                            continue

                        distance = (np.asarray([img_centre_unsorted['pedestrian'][0]] *
                                               len(img_centre_unsorted[object_keys]))
                                    - np.asarray(img_centre_unsorted[object_keys]))

                        distance = np.linalg.norm(distance, axis=1).reshape(-1, 1)
                        distance_sorted = sorted(distance)

                        for _, dist in enumerate(distance_sorted):
                            index = distance.tolist().index(dist.tolist())
                            img_centre[object_keys].append(img_centre_unsorted[object_keys][index])
                            img_features[object_keys].append(img_features_unsorted[object_keys][index])
                            bbox_location[object_keys].append(bbox_location_unsorted[object_keys][index])

                    node_features.append(img_features.copy())  # Path of the node features
                    bbox_location_seq.append(bbox_location.copy())  # BBox location
                    img_centre_seq.append(img_centre.copy())  # Bounding box centre location

                all_node_features_seq = []
                bbox_location_all_node = []
                img_centre_all_node = []

                count_dict = {'pedestrian': False,  # default should be one
                                  'vehicle': False,
                                  'traffic_light': False,
                                  'transit_station': False,
                                  'sign': False,
                                  'crosswalk': False,
                                  'ego_vehicle': False}

                for s in range(self.seq_len):

                    all_node_features = []
                    bbox_location_seq_all_node = []
                    img_centre_seq_all_node = []
                    # print(node_features[0])
                    for k in node_features[0]:
                        if k == 'pedestrian':
                            count_ped.append(len(node_features[s][k]))

                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['pedestrian']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['pedestrian']:
                                    all_node_features.append(0)

                        if k == 'vehicle':
                            count_veh.append(len(node_features[s][k]))

                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['vehicle']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['vehicle']:
                                    all_node_features.append(0)

                        if k == 'traffic_light':
                            count_traflig.append(len(node_features[s][k]))
                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['traffic_light']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['traffic_light']:
                                    all_node_features.append(0)

                        if k == 'transit_station':
                            count_transit.append(len(node_features[s][k]))
                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['transit_station']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['transit_station']:
                                    all_node_features.append(0)

                        if k == 'sign':
                            count_sign.append(len(node_features[s][k]))
                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['sign']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['sign']:
                                    all_node_features.append(0)

                        if k == 'crosswalk':
                            count_cross.append(len(node_features[s][k]))
                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['crosswalk']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])
                                elif len(node_features[s][k]) < self.node_info['crosswalk']:
                                    all_node_features.append(0)

                        if k == 'ego_vehicle':
                            for num, saving_nodes in enumerate(zip(node_features[s][k],
                                                                   bbox_location_seq[s][k],
                                                                   img_centre_seq[s][k])):

                                if num < self.node_info['ego_vehicle']:
                                    count_dict[k] = True
                                    all_node_features.append(saving_nodes[0])
                                    bbox_location_seq_all_node.append(saving_nodes[1])
                                    img_centre_seq_all_node.append(saving_nodes[2])

                    all_node_features_seq.append(all_node_features)
                    bbox_location_all_node.append(bbox_location_seq_all_node)
                    img_centre_all_node.append(img_centre_seq_all_node)

                for k in count_dict.keys():
                    if count_dict[k] == True:
                            count_object_samples[k] += 1



                self.feature_save_folder = './data/nodes_and_features/' + str(self.data_type)
                self.feature_save_path = os.path.join(self.feature_save_folder, str(i) + '.pkl')
                if not os.path.exists(self.feature_save_folder):
                    os.makedirs(self.feature_save_folder)
                with open(self.feature_save_path, 'wb') as fid:
                    pickle.dump((img_centre_all_node, all_node_features_seq, bbox_location_all_node), fid,
                                pickle.HIGHEST_PROTOCOL)


    def __getitem__(self, index):

        # crop only bounding boxes
        graph, adj_matrix, decoder_input, node_label, class_label = self.load_images_and_process(index)

        train_data = torch.from_numpy(graph), \
                     torch.from_numpy(adj_matrix), \
                     torch.from_numpy(decoder_input), \
                     torch.from_numpy(self.track['output'][index][0]), \
                     torch.from_numpy(node_label), \
                     torch.from_numpy(class_label)

        return train_data

    def anorm(self, p1, p2):
        NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if NORM == 0:
            return 0
        return 1 / (NORM)

    def get_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def get_path(self,
                 type_save='models',  # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder='./data/',
                 ind=1):

        assert (type_save in ['models', 'data'])
        if data_type != '':
            assert (any([d in data_type for d in ['images', 'features']]))
        if ind == 0:
            root = os.path.join(save_root_folder)
        else:
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


    def normalize_undigraph(self, A):

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)

        DAD = np.dot(np.dot(Dn, A), Dn)

        return DAD

    def load_images_and_process(self,
                                index,
                                visualise=False):

        with open(os.path.join(self.feature_save_folder, str(index) + '.pkl'), 'rb') as fid:
            try:
                img_centre_seq, node_features, bbox_location_seq = pickle.load(fid)
            except:
                img_centre_seq, node_features, bbox_location_seq = pickle.load(fid, encoding='bytes')

        max_nodes = self.max_nodes

        decoder_input = np.zeros((self.seq_len, max_nodes, len(bbox_location_seq[0][0])))
        graph = np.zeros((self.seq_len, max_nodes, 512, 7, 7))
        
        adj_matrix = np.zeros((self.seq_len, max_nodes, max_nodes))
        node_label = np.zeros((self.seq_len, max_nodes, max_nodes))
        class_label = np.zeros((self.seq_len, max_nodes))
        for s in range(self.seq_len):

            step = node_features[s]
            bbox_location = bbox_location_seq[s]

            img_cp_p = bbox_location[0]
            for h, stp in enumerate(step):
                if stp != 0:
                    with open(str(stp[0]), 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)

                        except:
                            img_features = pickle.load(fid, encoding='bytes')
                    node_label[s, h, h] = 1
                    class_label[s, h] = 1
                    img_features = np.squeeze(img_features)
                    graph[s, h, :] = img_features
                    decoder_input[s, h, :] = bbox_location[h]


                    adj_matrix[s, h, h] = 1

                    if self.structure != 'star' and self.structure != 'fully_connected':
                        print('Model excepts only "star" or "fully_connected" topology')
                        exit()

                    if self.structure == 'star' and h > 0:

                            adj_matrix[s, h, 0] = 1 
                            adj_matrix[s, 0, h] = 1 

                    elif self.structure == 'fully_connected':
                        # For fully connected
                        for k in range(h+1, len(step)):

                            adj_matrix[s, h, k] = 1
                            adj_matrix[s, k, h] = 1


            adj_matrix[s, :, :] = self.normalize_undigraph(adj_matrix[s, :, :])

        if visualise:
            self.visualisation(self.seq_len, node_features, img_centre_seq)

        return graph, adj_matrix, decoder_input, node_label, class_label

    def __len__(self):
        return self.num_examples

 