from utils.common import pad_data, shuffle_in_unison, check_ext_mem, check_ram_usage
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

#Cuts training data into batches
class CORE50Helper(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, x, y, t, train=True, transform=None):
        """
        Args:
            x: Data samples from core50 task
            y : "  "         labels   "   "     "
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.train = train
        self.task_id = t
        self.data = x
        self.targets = y
        # self.class_mapping = {c: i for i, c in enumerate(classes)}
        # self.class_indices = {}
        self.transform = transform

        # for cls in classes:
        #     self.class_indices[self.class_mapping[cls]] = []
        
        # if self.train:
        #     train_data = []
        #     train_labels = []
        #     train_tt = []  # task module labels
        #     train_td = []  # disctiminator labels

        #     for i in range(len(self.data)):
        #         if self.targets[i] in classes:
        #             train_data.append(self.data[i])
        #             train_labels.append(self.class_mapping[self.targets[i]])
        #             train_tt.append(t)
        #             train_td.append(t+1)
        #             self.class_indices[self.class_mapping[self.targets[i]]].append(i)
            
        #     self.train_data = np.array(train_data)
        #     self.train_labels = train_labels
        #     self.train_tt = train_tt
        #     self.train_td = train_td

        #Need to handle test-set case ****



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx,:,:,:]
        y = self.labels[idx,:,:,:]
        # tt = self.train_tt[idx]
        # td = self.train_td[idx]
        if(self.transform != None):
            x=self.transform(x)
            y = self.transform(y)
        
        return x, y #, tt, td
