import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, config):
        self.config = config
        self.n_class = len(self.config.new_class)
        self.label_connector = {}
        self.get_label_changer()
        self.set_data(self.config.minority_subsample_rate)
        self.get_rho()

    def get_label_changer(self):
        new_class_setting = self.config.new_class  # old classes to new class
        for new_label in new_class_setting.keys():
            for old_label in new_class_setting[new_label]:
                self.label_connector[old_label] = new_label

        for new_label, old_label in new_class_setting.items():
            print("\nNew label {} = old label".format(new_label), *old_label)

    def set_data(self, minority_subsample_rate=1):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # 1. Delete non-used class
        survived_class = list(self.label_connector.keys())
        self.x_train = x_train[np.isin(y_train, survived_class).squeeze()]
        self.x_test = x_test[np.isin(y_test, survived_class).squeeze()]
        y_train = y_train[np.isin(y_train, survived_class).squeeze()].reshape(-1, 1)
        y_test = y_test[np.isin(y_test, survived_class).squeeze()].reshape(-1, 1)

        # 2. Change from old label to new label
        self.y_train, self.y_test = -np.ones_like(y_train), -np.ones_like(y_test)
        for i, old in enumerate(y_train.squeeze()):
            self.y_train[i, 0] = self.label_connector[old]
        for i, old in enumerate(y_test.squeeze()):
            self.y_test[i, 0] = self.label_connector[old]

        # 3. Minor class subsampling for decrease imbalance ratio
        if minority_subsample_rate < 1:
            # decrease the number of minority class
            nums_cls = self.get_class_num()
            delete_indices = set()
            for minor_cl in self.config.minor_classes:
                num_cl = nums_cls[minor_cl]
                idx_cl = np.where(self.y_train == minor_cl)[0]
                delete_idx = np.random.choice(idx_cl, int(num_cl * (1 - minority_subsample_rate)), replace=False)
                delete_indices.update(delete_idx)
            survived_indices = set(range(len(self.y_train))).difference(delete_indices)
            self.y_train = self.y_train[list(survived_indices)]
            self.x_train = self.x_train[list(survived_indices)]
        print("\nNumber of each class.")
        for cl_idx, cl_num in enumerate(self.get_class_num()):
            print("\t- Class {} : {}".format(cl_idx, cl_num))
        print("\nImbalance ratio compared to major class.")
        for cl_idx, cl_ratio in enumerate(self.get_class_num() / max(self.get_class_num())):
            print("\t- Class {} : {:.3f}".format(cl_idx, cl_ratio))

        # 4. Data normalization
        image_mean = np.array([self.x_train[..., i].mean() for i in range(3)])
        self.x_train = (self.x_train - image_mean) / 255
        self.x_test = (self.x_test - image_mean) / 255

    def get_class_num(self):
        # get number of all classes
        _, nums_cls = np.unique(self.y_train, return_counts=True)
        return nums_cls

    def get_rho(self):
        """
        In the two-class dataset problem, this paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_class_num()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))


class CassavaLeafDataset:
    def __init__(self, image_size = (512, 512), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.reading_csv("../Deep-Reinforcement-Learning-on-Imbalanced-Data/cassava-leaf-disease-classification/train_images/", "../Deep-Reinforcement-Learning-on-Imbalanced-Data/cassava-leaf-disease-classification/train.csv")
        self.create_dataset()

        self.get_rho()
        self.get_minority_classes()

    def reading_csv(self, folder_path, file_path):
        df = pd.read_csv(file_path) # Load train image file names and each label data
        df["filepath"] = folder_path + df["image_id"] # Create path by adding folder name and image name for load images easily
        df = df.drop(['image_id'],axis=1) # Drop image names which is useless.
        self.X = df.drop(columns=["label"])
        self.y = df['label']

    
    def imbalance_the_data(self):
        pass

    def load_image_and_label_from_path(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        return img, label
    
    def create_dataset(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.2)
        
        training_data = tf.data.Dataset.from_tensor_slices((x_train.filepath.values, y_train))
        testing_data = tf.data.Dataset.from_tensor_slices((x_test.filepath.values, y_test))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        training_data = training_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
        testing_data = testing_data.map(self.load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

        self.training_data_batches = training_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        self.testing_data_batches = testing_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        
        # Create a TensorFlow session
        with tf.compat.v1.Session() as sess:
            train_iterator = tf.compat.v1.data.make_one_shot_iterator(self.training_data_batches)
            train_next_element = train_iterator.get_next()
        
            while True:
                try:
                    features, labels = sess.run(train_next_element)
                    for i in range(len(labels)):
                        self.x_train.append(features[i]/255)
                        self.y_train.append((labels[i],))    
                except tf.errors.OutOfRangeError:
                    break
            
            test_iterator = tf.compat.v1.data.make_one_shot_iterator(self.testing_data_batches)
            test_next_element = test_iterator.get_next()
            
            while True:
                try:
                    features, labels = sess.run(test_next_element)
                    for i in range(len(labels)):
                        self.x_test.append(features[i]/255)
                        self.y_test.append((labels[i],))
                        
                except tf.errors.OutOfRangeError:
                    break
        

    def get_class_num(self):
        # get number of all classes
        _, nums_cls = np.unique(self.y, return_counts=True)
        print("No of total samples in dataset and their distribution: ", np.unique(self.y, return_counts=True))
        
        return nums_cls

    def get_minority_classes(self):
        label, label_count = np.unique(self.y, return_counts=True)
        print("Label is ", label)
        labels_with_counts = {}
        for i in range(len(label)):
            labels_with_counts[label[i]] = label_count[i]
        print("Labels with count",labels_with_counts)
        labels_with_counts = sorted(labels_with_counts.items())

        # We are going to get 35% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        no_of_minority_classes_to_get = int(np.round(len(label) * 0.35))

        self.minority_classes = []
        for i in range(no_of_minority_classes_to_get):
            self.minority_classes.append(labels_with_counts[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, research paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_class_num()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))


class PersonalityDataset:
    def __init__(self, batch_size=100):

        self.batch_size = batch_size
        self.create_dataset()
        self.get_rho()
        self.get_minority_classes()

    def create_dataset(self):
        df = pd.read_csv("./16P/16P.csv", encoding='cp1252')
        
        df = df.dropna()

        self.X = df.drop(["Personality", "Response Id"], axis = 1)
        self.y = df["Personality"]

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        
        # self.y = tf.keras.utils.to_categorical(self.y)

        self.unique_labels, self.label_counts = np.unique(self.y, return_counts=True)
        
        x_train, x_test, y_train, y_test = train_test_split(self.X.values, self.y, random_state=42, test_size=0.2)
        
        
        # We are going to get 25% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        self.no_of_minority_classes_to_get = int(np.round(len(np.unique(y_train)) * 0.25))
        
        # Specify the percentage of label 2 data to remove
        percentage_to_remove = 80
        for class_to_remove in range(self.no_of_minority_classes_to_get):
            indices_to_remove = np.unique(np.where(y_train == class_to_remove)[0])
            # Calculate the number of samples to remove
            num_samples_to_remove = int(percentage_to_remove / 100 * len(indices_to_remove))

            # Randomly select indices to remove
            indices_to_remove = np.random.choice(indices_to_remove, num_samples_to_remove, replace=False)
            # Remove the selected samples
            percentage_to_remove -= 10

            # Remove the selected samples
            x_train = np.delete(x_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            percentage_to_remove -= 10

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = [[y_val] for y_val in y_train]
        self.y_test = [[y_val] for y_val in y_test]

        self.length_of_dataset = len(x_train)
        
    def get_labels_counts(self):
        self.unique_labels, self.label_counts = np.unique(self.y_train, return_counts=True)
        
        return self.label_counts
        
    def get_minority_classes(self):
        """
        We are going to get minority label classes from the dataset using this function
        """
        
        unique_labels, label_counts = np.unique(self.y_train, return_counts=True)
        print("Label is ", unique_labels)
        labels_with_counts = {}


        for i in range(len(unique_labels)):
            labels_with_counts[unique_labels[i]] = label_counts[i]
        
        print("Labels with count",labels_with_counts)
        labels_with_counts = sorted(labels_with_counts.items())

        # We are going to get 35% minority classes from total classes i-e if there are total 6 classes then we will only set 2 classes as minority classes
        no_of_minority_classes_to_get = int(np.round(len(unique_labels) * 0.35))

        self.minority_classes = []
        for i in range(no_of_minority_classes_to_get):
            self.minority_classes.append(labels_with_counts[i][0])

    def get_rho(self):
        """
        In the two-class dataset problem, research paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_labels_counts()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))