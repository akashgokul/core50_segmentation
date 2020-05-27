from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd

#Cuts training data into batches
class CORE50(Dataset):

    """Helper Dataset for CORE50 Data"""

    def __init__(self, x, t, num_tasks, y = None, split= 'train', task_type= 'classify', classes=0, transform=None):
        """
        Args:
            x: (Numpy Array or List) Data 
            y : (Optional, Dict) Ground Truth (e.g. Labels, Segmentation Masks, etc.)
            t: (Int) Task Number
            num_tasks: (Int) Total Number of Tasks in scenario
            split: (String) Denotes whether this is a test (split = 'test') or training (split = 'train') dataset 
            task_type: (String) Denotes task type (e.g. 'classify' = classification, 'detection' = object detection, 'segment' = object segmentation)
            classes: (Optional, List) Number of classes in each task (only applicable for classification or detection)
            transform: (Optional, Torch Transform) Torch Transform to apply to data
        """

        #Checking arguments are valid
        assert isinstance(t,int)
        assert isinstance(num_tasks,int)
        task_type = task_type.lower()
        assert task_type in ['classify','detect','segment']
        split = split.lower()
        assert split in ['train','test']

        #Initializing Dataset
        self.task_id = t
        self.data = x
        self.targets = y
        self.split = split
        self.num_samples = len(self.data) if isinstance(self.data,list) else self.data.shape[0]
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.transform = transform

        self.tt = [self.task_id for _ in range(len(self.data))]
        self.td = [self.task_id + 1 for _ in range(len(self.data))]
        self.inputsize = self.data[0].shape

        #TODO: RECOMPUTE JUST IN CASE, THESE VALUES ARE FROM CORE50 CHALLENGE
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229,0.224,0.225]

        #TODO: Need to handle obj detect
        if(self.task_type in ['classify', 'detect']):
            
            if not isinstance(classes, list):
                classes_ = [classes]

            self.class_mapping = {c: i for i, c in enumerate(classes_)}
            self.class_indices = {}

            for cls in classes_:
                self.class_indices[self.class_mapping[cls]] = []

            data = []
            labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes_:
                    data.append(self.data[i])
                    labels.append(self.class_mapping[self.targets[i]])


            self.data = np.array(data)
            self.labels = labels

            self.num_classes = len(set(labels))
            # assert self.num_classes == len(classes_), "{}! Number of classes does not match with 100".format(
            #     self.num_classes)
        
        #Handles obj segmentation 
        else:
            self.labels = self.targets




    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        tt = self.tt[idx]
        td = self.td[idx]

        if(self.transform != None):
            x = self.transform(x)
        
        if(self.split == 'test'):
            return x, tt, td
        else:
            return x, y, tt, td
