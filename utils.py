import torch 
import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import datetime
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import logging
logging.basicConfig(level=logging.INFO)

# def preprocess(dataset, transforms):
#     dataset = copy.copy(dataset) #shallow copy
#     for transform in transforms:
#         dataset['data'] = transform(dataset['data'])
#     return dataset

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms): #reversed
        dataset.data = transform(dataset.data)
    return dataset
 
class AccuracyMeter(object):
    
    def __init__(self, name, model_dir):
        self.name = name
        self.file = open(f'{model_dir}/{name}_stats.txt', 'a')
        self.reset()
    
    def reset(self):
        self.stats = {'acc': [], 'loss': []}
        self.cnt = 0
        self.time = 0.0
    
    def update(self, stats, time, n=1):
        self.stats['acc'].append(stats['acc'])
        self.stats['loss'].append(stats['loss'])
        self.cnt += n
        self.time += time
        logging.info(f"{self.name}_acc: {round(stats['acc'], 2)}, time for step{self.cnt}: {round(time, 2)}, total {self.name} time: {round(self.time, 2)}")
        self.file.write(f"{self.name}_acc: {round(stats['acc'], 2)}, time for step{self.cnt}: {round(time, 2)}, total {self.name} time: {round(self.time, 2)}\n")
    def get(self):
        return self.stats, self.cnt, self.time
    def plot(self, out_dir):
        out_dir = os.path.join(out_dir, 'plots')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for key in self.stats.keys():
            plt.plot(self.stats[key])
            plt.title(f'{self.name}_{key}')
            plt.ylabel(f'{self.name}_{key}')
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, self.cnt, 5))
            plt.grid(True)
            plt.savefig(f"{os.path.join(out_dir, f'{self.name}_{key}')}.png")
            plt.close()
        self.file.close()
        
def write_to_file(save_dict, file_name):
    fo = open(file_name, "a")
    for k, v in save_dict.items():
        fo.write(str(k) + ': '+ str(v) + '\t')
    fo.write('\n')
    fo.close()
    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True)
#     print(trainset.data[0].shape)
#     # indices = list(range(int(split*len(trainset))))
#     # valid_indices =  list(range(int(split*len(trainset)), len(trainset)))
#     logging.info(f"Training size= {len(trainset)}")
#     # training_sampler = SubsetRandomSampler(indices)
#     # valid_sampler = SubsetRandomSampler(valid_indices)
#     trainloader = torch.utils.data.DataLoader(dataset=transform.Transform(trainset, train_transform),
#                                             batch_size=batch_size) #,
#                                             #sampler=training_sampler) 

#     # validloader = torch.utils.data.DataLoader(dataset=transform.Transform(trainset, test_transform), 
#     #                                         batch_size=batch_size, 
#     #                                         sampler=valid_sampler)                
    
#     testloader = torch.utils.data.DataLoader(transform.Transform(testset, test_transform),
#                                             batch_size=batch_size,
#                                             shuffle=False)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def weights_init_uniform(m):
        classname = m.__class__.__name__
        print(classname)
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.fill_(0.0)
            # m.bias.data.fill_(0)
        elif classname.find('Conv') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.fill_(0.0)
            # m.bias.data.fill_(0)


# class LayerTimer:
#     def __init__(self, name):
#         self.name = name
#         self.total_time = 0
#         self.timer = None
#         self.times = list()
    
#     def start_time(self):
#         self.timer = datetime.datetime.now()

#     def step(self):
#         time = -(self.timer - datetime.datetime.now()).total_seconds()
#         self.times.append(time)
#         self.timer = None
    
#     def get_last_time(self):
#         return self.times[-1]
    
#     def get_stats(self):
#         # with(f"{name}_stats.txt", 'w') as f:
#         #     f.write('Mean:' + str(np.mean(self.times)) + '\tTotal Time:'+ str(np.sum(self.times))) 
#         return np.mean(self.times), np.sum(self.times)


    # conv_stats = list()
    # bn_stats = list()
    # activ_stats = list()
    # linear_stats = list()
    # for module in model.modules():
    #     if type(module).__name__ == 'Network':
    #         linear_stats.append(module.linear_timer.get_stats())
    #     elif type(module).__name__ == 'conv_bn_self':
    #         conv_stats.append(module.conv_timer.get_stats())
    #         bn_stats.append(module.bn_timer.get_stats())
    #         activ_stats.append(module.activ_timer.get_stats())

    # conv_stats = np.array(conv_stats) 
    # ret_dict['total_conv_time'] = np.sum(conv_stats, axis=0)[1]
    # ret_dict['total_bn_time'] = np.sum(bn_stats, axis=0)[1]
    # ret_dict['total_activ_time'] = np.sum(activ_stats, axis=0)[1]
    # ret_dict['total_linear_time'] = np.sum(linear_stats, axis=0)[1]


class Vocab(object):

    PAD = 0
    EOS = 1
    UNK = 3
    BOS = 2

    def __init__(self, dict_path, max_n_words=-1):

        with open(dict_path) as f:
            _dict = json.load(f)

        # Word to word index and word frequence.
        self._token2id_feq = self._init_dict()

        N = len(self._token2id_feq)

        for ww, vv in _dict.items():
            if isinstance(vv, int):
                self._token2id_feq[ww] = (vv + N, 0)
            else:
                self._token2id_feq[ww] = (vv[0] + N, vv[1])

        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])

        self._max_n_words = max_n_words

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<PAD>": (Vocab.PAD, 0),
            "<UNK>": (Vocab.UNK, 0),
            "<EOS>": (Vocab.EOS, 0),
            "<BOS>": (Vocab.BOS, 0)
                }

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return Vocab.UNK

    def id2token(self, id):

        return self._id2token[id]

    @staticmethod
    def special_ids():

        return [0, 1, 2]