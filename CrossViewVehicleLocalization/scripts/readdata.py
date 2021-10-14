import numpy as np
from scipy.spatial import KDTree
import copy
import random
import cv2
import scipy

class InputData:
    
    def __init__(self, radius):
        print('hypothesis coarse localization prior:', radius)        
        self.radius = radius
        self.gt_radius = 5 
        self.gt_radius_val = 5
        self.sig = 10
        self.image_root = '/local/zxia/datasets/Oxford_5m_sampling/' # replace this with the path to the dataset
        self.datasplit_root = '/local/zxia/experiments/Visual-Localization-with-Spatial-Prior/CrossViewVehicleLocalization/datasplits/' # replace this with the path to the datasplit files
        
        # load the training, validation, and test set
        trainlist = []
        with open(self.datasplit_root + 'training.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                trainlist.append(content.split(" "))
                
        vallist = []
        with open(self.datasplit_root + 'validation.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                vallist.append(content.split(" "))
                
        testlist1 = []
        with open(self.datasplit_root + 'test1.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                testlist1.append(content.split(" "))
                
        testlist2 = []
        with open(self.datasplit_root + 'test2.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                testlist2.append(content.split(" "))
                
        testlist3 = []
        with open(self.datasplit_root + 'test3.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                testlist3.append(content.split(" "))
                
        testlist = testlist1 + testlist2 + testlist3
                
        self.trainList = trainlist
        self.trainNum = len(trainlist)
        trainarray = np.array(trainlist)
        self.trainUTM = np.transpose(trainarray[:,2:].astype(np.float64))
        
        self.valList = vallist
        self.valNum = len(vallist)
        valarray = np.array(vallist)
        self.valUTM = np.transpose(valarray[:,2:].astype(np.float64))
        
        self.testList = testlist
        self.testNum = len(testlist)
        testarray = np.array(testlist)
        self.testUTM = np.transpose(testarray[:,2:].astype(np.float64))
        
        fulllist = vallist+testlist+trainlist
        self.fullList = fulllist
        self.fullNum = len(fulllist)
        fullarray = np.array(fulllist)
        self.fullUTM = np.transpose(fullarray[:,2:].astype(np.float64))
        self.UTMTree = KDTree(np.transpose(self.fullUTM))
        self.fullIdList = [*range(0,self.fullNum,1)]
        self.IdList_to_use = []
        
        print('number of satellite images', self.fullNum)    
        print('number of ground images in training set', self.trainNum)    
        print('number of ground images in validation set', self.valNum) 
        print('number of ground images in test set', self.testNum)

        print('Storing the index of nearby images for all satellite images. This might take a while')
        UTM_transposed = np.transpose(self.fullUTM)
        UTMTree = KDTree(UTM_transposed)
        
        self.neighbor_gt_train = {} # store the index of positive samples
        self.neighbor_gt_val = {}
        self.neighbor_negative_samples_train = {} # store the index of nearby negative samples 
        self.neighbor_negative_samples_val = {}
        for i in range(self.fullNum):
            center_UTM = np.transpose(self.fullUTM[:,i])
            idx_gt_train = UTMTree.query_ball_point(center_UTM,r=self.gt_radius, p=2)
            self.neighbor_gt_train.update({str(i):idx_gt_train})
            
            candidates = UTMTree.query_ball_point(center_UTM,r=self.radius, p=2)
            
            idx_negative_samples_train = [x for x in candidates if x not in idx_gt_train]
            self.neighbor_negative_samples_train.update({str(i):idx_negative_samples_train})
            
            idx_gt_val = UTMTree.query_ball_point(center_UTM,r=self.gt_radius_val, p=2)
            self.neighbor_gt_val.update({str(i):idx_gt_val})
            idx_negative_samples_val = [x for x in candidates if x not in idx_gt_val]
            self.neighbor_negative_samples_val.update({str(i):idx_negative_samples_val})
            
        self.__cur_id = 0 
        self.__cur_test_id = 0
        
        

    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.fullNum:
            self.__cur_test_id = 0
            return None, None, None
        elif self.__cur_test_id + batch_size >= self.fullNum:
            batch_size = self.fullNum - self.__cur_test_id

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 154, 231, 3], dtype=np.float32)
        batch_utm = np.zeros([batch_size, 2], dtype=np.float64)
        
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.image_root + self.fullList[img_idx][1])
             
            if img is None:
                print('InputData::next_pair_batch: read fail: %s' % (self.fullList[img_idx][1]))
                continue
                
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.image_root + self.fullList[img_idx][0])

            if img is None:
                print('InputData::next_pair_batch: read fail: %s' % (self.fullList[img_idx][0]))
                continue
            img = cv2.resize(img, (231, 154), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

            batch_utm[i,0] = self.fullUTM[0, img_idx]
            batch_utm[i, 1] = self.fullUTM[1, img_idx]

        self.__cur_test_id += batch_size


        return batch_sat, batch_grd, batch_utm


    def next_pair_batch(self, batch_size):
        if self.__cur_id == 0:
            self.IdList_to_use = copy.deepcopy(self.fullIdList) # a fresh list is created every epoch
            for i in range(20):
                random.shuffle(self.IdList_to_use)          

        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 154, 231, 3], dtype=np.float32)
        batch_utm_sat = np.zeros([batch_size, 2], dtype=np.float64)
        batch_utm_grd = np.zeros([batch_size, 2], dtype=np.float64)
        
        batch_idx = 0
        i = 1
        empty_grd = []
        while True:
            if self.__cur_id + batch_size >= self.fullNum:
                self.__cur_id = 0 
                return None, None, None, None, None
            
            if batch_idx >= batch_size:
                break
            
            if self.__cur_id + i >= len(self.IdList_to_use):
                # go to next center image if there is no enough nearby images in the remaining list
                self.__cur_id += 1
                batch_idx = 0
                i = 1
                continue
            
            if batch_idx == 0:
            # Load the center image and its coordinates
                img_idx = self.IdList_to_use[self.__cur_id]
                
                # Randomly pick an image in the pre-denfined area as ground truth
                gt_sat_idx_list = self.neighbor_gt_train[str(img_idx)]
                gt_sat_idx = random.choice(gt_sat_idx_list)
                
                # Get the indexes of nearby negatives for the current center image
                candidates = self.neighbor_negative_samples_train[str(img_idx)]
                
                # If there is not enough nearby images, then move to the next center image
                if len(candidates) < batch_size-1:
                    print('There is no enough nearby images for current query')
                    self.__cur_id += 1
                    continue
                    
                # satellite, satellite image located at the circle centered at the ground image with gt_radius
                img = cv2.imread(self.image_root + self.fullList[gt_sat_idx][1])
                if img is None:
                    print('InputData::next_pair_batch: read fail: %s' % (self.fullList[gt_sat_idx][1]))
                    self.__cur_id += 1
                    continue
                    
                img = img.astype(np.float32)
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                rand_rotate = random.randint(0, 4) * 90
                rot_matrix = cv2.getRotationMatrix2D((128, 128), rand_rotate, 1)
                img = cv2.warpAffine(img, rot_matrix, (256, 256))
                batch_sat[batch_idx, :, :, :] = img
                
                # ground
                if self.fullList[img_idx] in self.trainList:
                    img = cv2.imread(self.image_root + self.fullList[img_idx][0])
                    if img is None:
                        print('InputData::next_pair_batch: read fail: %s' % (self.fullList[img_idx][0]))
                        self.__cur_id += 1
                        continue
                    img = cv2.resize(img, (231, 154), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32)
                    img[:, :, 0] -= 103.939  # Blue
                    img[:, :, 1] -= 116.779  # Green
                    img[:, :, 2] -= 123.6  # Red
                    batch_grd[batch_idx, :, :, :] = img
                else:
                    empty_grd.append(batch_idx)
                
                # coordinates
                batch_utm_sat[batch_idx,0] = self.fullUTM[0, gt_sat_idx]
                batch_utm_sat[batch_idx, 1] = self.fullUTM[1, gt_sat_idx]
                batch_utm_grd[batch_idx,0] = self.fullUTM[0, img_idx]
                batch_utm_grd[batch_idx, 1] = self.fullUTM[1, img_idx]
                
                batch_idx += 1
            else:
            ## Load other neaby images into the batch
                if self.IdList_to_use[self.__cur_id + i] in candidates:
                    img_idx = self.IdList_to_use[self.__cur_id + i]
                    # Remove this index from the current list
                    del self.IdList_to_use[self.__cur_id + i]
                    
                    # satellite
                    gt_sat_idx_list = self.neighbor_gt_train[str(img_idx)]
                    gt_sat_idx = random.choice(gt_sat_idx_list)
                    img = cv2.imread(self.image_root + self.fullList[gt_sat_idx][1])
                    if img is None:
                        print('InputData::next_pair_batch: read fail: %s' % (self.fullList[gt_sat_idx][1]))
                        continue
                    img = img.astype(np.float32)
                    img[:, :, 0] -= 103.939  # Blue
                    img[:, :, 1] -= 116.779  # Green
                    img[:, :, 2] -= 123.6  # Red
                    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    rand_rotate = random.randint(0, 4) * 90
                    rot_matrix = cv2.getRotationMatrix2D((128, 128), rand_rotate, 1)
                    img = cv2.warpAffine(img, rot_matrix, (256, 256))
                    img = img.astype(np.float32)
                    batch_sat[batch_idx, :, :, :] = img
                    
                    # ground
                    if self.fullList[img_idx] in self.trainList:
                        img = cv2.imread(self.image_root + self.fullList[img_idx][0])
                        if img is None:
                            print('InputData::next_pair_batch: read fail: %s' % (self.fullList[img_idx][0]))
                            continue
                        img = cv2.resize(img, (231, 154), interpolation=cv2.INTER_AREA)
                        img = img.astype(np.float32)
                        img[:, :, 0] -= 103.939  # Blue
                        img[:, :, 1] -= 116.779  # Green
                        img[:, :, 2] -= 123.6  # Red
                        batch_grd[batch_idx, :, :, :] = img
                    else:
                        empty_grd.append(batch_idx)
                        
                    batch_utm_sat[batch_idx, 0] = self.fullUTM[0, gt_sat_idx]
                    batch_utm_sat[batch_idx, 1] = self.fullUTM[1, gt_sat_idx]
                    batch_utm_grd[batch_idx,0] = self.fullUTM[0, img_idx]
                    batch_utm_grd[batch_idx, 1] = self.fullUTM[1, img_idx]
                    
                    batch_idx += 1
                else:
                    i += 1
     
        self.__cur_id += 1

        distance_matrix = scipy.spatial.distance_matrix(batch_utm_sat, batch_utm_grd)

        useful_pairs = (distance_matrix<=self.radius).astype(np.int)*(distance_matrix>=self.gt_radius).astype(np.int)
        np.fill_diagonal(useful_pairs, 0)

        useful_pairs_s2g = copy.deepcopy(useful_pairs)
        useful_pairs_g2s = copy.deepcopy(useful_pairs)
        # mark the pairs contain a non-training ground image. Those pairs will not contribute to the loss
        for i in empty_grd:
            useful_pairs_s2g[:,i] = 0
            useful_pairs_s2g[i,:] = 0
            useful_pairs_g2s[:,i] = 0 
                
        return batch_sat, batch_grd, distance_matrix, useful_pairs_s2g, useful_pairs_g2s
    
    def get_dataset_size(self):
        return self.trainNum

    def get_test_dataset_size(self):
        return self.valNum
    
    def get_full_dataset_size(self):
        return self.fullNum

    def reset_scan(self):
        self.__cur_test_id = 0


    