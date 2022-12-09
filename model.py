
from skimage.color import rgb2gray
import skimage
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import classification_report
import cv2
from tensorflow import keras
import math
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
no_of_train_images = len(train_images)

no_of_test_images = len(test_images)

descriptors = []
keypoints = []
index_images = []
SIFT = cv2.xfeatures2d.SIFT_create()

for i in range(no_of_train_images):
  gray_image = rgb2gray(train_images[i]);
  kypts,desc = SIFT.detectAndCompute(gray_image,None);
  for j in range(len(kypts)):
    descriptors.append(desc[j]);
    keypoints.append(kypts[j])
    index_images.append(i)

desc = np.asarray(desc)

def CreateVisualDictionary(clusters,iterations,descriptors):
  cluster_center = {}
  div_descriptors = {}
  len_desc = len(descriptors)
  tolerance = 0.001


  for i in range(clusters):
    cluster_center[i] = descriptors[i]


  for num_iteration in range(iterations):
    
    points_to_clusters = {}
    index_classifications={}
    div_descriptors ={i:[] for i in range(clusters)}
    points_to_clusters ={i:-1 for i in range(clusters)}
    index_classifications ={i:[] for i in range(clusters)}
   
    index = 0
    for desc in descriptors:
      distances = []
      for i in range(clusters):
        distances.append( np.linalg.norm(desc-cluster_center[i]))
      division = distances.index(min(distances))
      div_descriptors[division].append(desc)
      index_classifications[division].append(index)
      points_to_clusters[index] = division
      index+=1

    old_centroids = (cluster_center)
    for division in div_descriptors:
      cluster_center[division] = np.average(div_descriptors[division],axis=0)
    
    optimized = True
    for c in cluster_center:
      old_val = old_centroids[c]
      new_val = cluster_center[c]
      if(np.sum((new_val-old_val)/new_val*100) > tolerance):
        optimized = False
    if(optimized):
      return cluster_center,div_descriptors,index_classifications,points_to_clusters
      break

# cluster ind stores centroid 
def division_closest(cluster_center,feature_set):
  distances = []
  for each_center in cluster_center:
      distances.append(np.linalg.norm(feature_set-cluster_center[each_center]))
  division = distances.index(min(distances))
  return division

def ComputeHistogram(closest_cluster,features,len_features):
  histogram = [0 for i in closest_cluster]
  for i in range(len_features):
    feature = features[i]
    division_closest1 = division_closest(closest_cluster,feature)
    histogram[division_closest1]+=1
  histogram = np.asarray(histogram)
  return histogram

def calculate_elbow(points_to_clusters,centroids,descriptors):
  pts= len(descriptors)
  distance = [math.pow(np.linalg.norm(descriptors[i]-centroids[points_to_clusters[i]]),2) for i in (range(pts))]
  elbow_index = np.sum(distance)
  return elbow_index

elbow_graph = {i:0 for i in range(1,999)}
test_k_for = 100
for i in range(1,test_k_for,10):
  if(elbow_graph[i]==0):
    cluster_center,div_descriptors,index_classifications,points_to_clusters  = CreateVisualDictionary(i,1,descriptors)
    elbow_graph[i] = calculate_elbow(points_to_clusters,cluster_center,descriptors)

def plot_elbow(elbow_graph,test_k_for):
  fig = plt.figure(figsize = (10, 5))
  temp = {}
  for i in range(1,test_k_for,10):
    temp[i] = elbow_graph[i]
  plt.plot(list(temp.keys()), list(temp.values()), color ='maroon')
  
  plt.xlabel("K")
  plt.ylabel("WSS")
  plt.title("Analysis")
  plt.show()

plot_elbow(elbow_graph,test_k_for)

"""## change here

"""

cluster_center,div_descriptors,index_classifications,points_to_clusters = CreateVisualDictionary(32,100,descriptors)

nearest_descriptor_index = {i:[] for i in cluster_center}
def find_nearest_descriptor(centroids,index_classifications):
  for centroid in centroids:
    distances =  [np.linalg.norm(div_descriptors[centroid][i]-centroids[centroid]) for i in (range(len(div_descriptors[centroid])))]
    desc_index = distances.index(min(distances))
    nearest_descriptor_index[centroid] = index_classifications[centroid][desc_index]
    # nearest_descriptor_index[centroid] = descriptors[index_classifications[centroid][desc_index]]
  
find_nearest_descriptor(cluster_center,index_classifications)

"""## paste into a text file

"""

nearest_descriptor_index

def plot_histogram(histogram,cluster_center):
  fig = plt.figure(figsize = (10, 5))
  plt.xlabel("division no.")
  plt.ylabel("feq")
 
  plt.bar([i for i in cluster_center], [histogram[i] for i in cluster_center], color ='green',width = 0.1)
  
  plt.title("histogram")
  plt.show()

# Compute cossine similarity
def MatchHistogram(feature1, feature2):
  dot_pro = np.dot(feature1,feature2)
  num = np.sum(dot_pro)
  deno = np.linalg.norm(feature1)*np.linalg.norm(feature2)
  if(deno==0):
    return 0
  return num/deno

pred = np.full(no_of_test_images,-1)

feature_train_vectors = {}
feature_test_vector = {}
for i in range(no_of_train_images):
  feature_train_vectors[i] = []
for i in range(no_of_test_images):
  feature_test_vector[i] = []

testing_for = no_of_test_images

def ComputeFeatureVectors(no_of_train_images,testing_for):
  for i in range(no_of_train_images):
    if(feature_train_vectors[i]==[]):
      gray_image = rgb2gray(train_images[i]);
      kypts,desc = SIFT.detectAndCompute(gray_image,None)
      feature_train_vectors[i] = ComputeHistogram(cluster_center,desc,len(kypts))
  for i in range(testing_for):
    if(feature_test_vector[i]==[]):
      gray_image = rgb2gray(test_images[i]);
      kypts,desc = SIFT.detectAndCompute(gray_image,None)
      feature_test_vector[i] = ComputeHistogram(cluster_center,desc,len(kypts))


ComputeFeatureVectors(no_of_train_images,testing_for)

def predict(testing_for):
  for i in range(testing_for):
    if(pred[i]==-1):
      test_vec = feature_test_vector[i]
      ind = -1
      cossine = [MatchHistogram(feature_train_vectors[j],test_vec) for j in range(no_of_train_images)]
      ind = cossine.index(max(cossine))
      pred[i] = train_labels[ind]

predict(testing_for)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def calculate_overall_accuracy(predictions,test_labels,test_for):
  correct  = 0;
  total = test_for
  for i in range(test_for):
    if(predictions[i]==test_labels[i]):
      correct+=1
  return (correct/total)*100

pred[:testing_for]

test_labels[:testing_for]

#@title Default title text

y_true = test_labels[:testing_for]
y_predict = pred[:testing_for]
print(classification_report(y_true,y_predict,target_names = class_names))
print("Overall accuracy => ")
print(accuracy_score(y_true,y_predict))



