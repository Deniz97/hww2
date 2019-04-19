import glob
import cv2
import numpy as np
import _pickle as pc
import os
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier as KNN
from random import shuffle
from random import choice

NUMBER_OF_IMAGES = 6000
to_write = {} #for logging purposes

SIFT_TYPE = "dense_sift" # "sift" or "dense_sift"
DENSE_STEP_SIZE = 5 #step size for dense sift

_nfeatures = 0  #default 0
_nOctaveLayers = 3  #default 3,
_contrastThreshold = 0.04   #default 0.04,
_edgeThreshold = 10 #default 10
_sigma = 1.6    #default 1.6 


_init_size = 300 #3*batch_size
_n_init = 10 #default 10
_reassignment_ratio = 0.01 #default 0.01
_batch_size = 10000 #default 100

K_MEANS_K = 128 # You think this is const? Cause it certainly is not.
KNN_K = 16

np.random.seed(7) #lol random

to_write["nfeatures"] = _nfeatures
to_write["OctaveLayers"] = _nOctaveLayers
to_write["contrastThreshold"] = _contrastThreshold
to_write["edgeThreshold"] = _edgeThreshold
to_write["sigma"] = _sigma
to_write["init_size"] = _init_size
to_write["n_init"] = _n_init
to_write["reassignment_ratio"] = _reassignment_ratio
to_write["batch_size"] = _batch_size
to_write["step-size"] = DENSE_STEP_SIZE


####### joke aside, dont go below, you have been warned

def generate_random(a,b,percentage=0,default=-1,intmi=False):
    #returns default with a change of percentage
    if( np.random.ranf() > percentage ):
        retval= (b-a) * np.random.ranf() + a
    else:
        retval = default

    if(intmi==True):
        return int(retval)

    return retval

def default_parameters():
    global _nfeatures
    global _nOctaveLayers
    global _contrastThreshold
    global _edgeThreshold
    global _sigma
    global _batch_size
    global _init_size
    global _n_init
    global _reassignment_ratio
    global K_MEANS_K
    global DENSE_STEP_SIZE
    
    _nOctaveLayers = 3  #default 3,
    _batch_size = 100 #default 100
    _init_size = 3000 #3*batch_size
    _n_init = 10 #default 10
    _reassignment_ratio =  0.01 #default 0.01
    _contrastThreshold = 0.04   #default 0.04,
    _edgeThreshold = 10 #default 10
    _sigma = 1.6 #default 1.6 
    if SIFT_TYPE == "sift":
        K_MEANS_K = 128
    elif SIFT_TYPE=="dense_sift":
        DENSE_STEP_SIZE = 10 #default 10
        K_MEANS_K = 256
    else:
        print("unrecognized sift type")
        exit()
    
    global to_write
    to_write["nfeatures"] = _nfeatures
    to_write["OctaveLayers"] = _nOctaveLayers
    to_write["contrastThreshold"] = _contrastThreshold
    to_write["edgeThreshold"] = _edgeThreshold
    to_write["sigma"] = _sigma
    to_write["init_size"] = _init_size
    to_write["n_init"] = _n_init
    to_write["reassignment_ratio"] = _reassignment_ratio
    to_write["batch_size"] = _batch_size
    to_write["step-size"] = DENSE_STEP_SIZE

def good_paremeters():
    global _nfeatures
    global _nOctaveLayers
    global _contrastThreshold
    global _edgeThreshold
    global _sigma
    global _batch_size
    global _init_size
    global _n_init
    global _reassignment_ratio
    global K_MEANS_K
    global DENSE_STEP_SIZE
    
    _nOctaveLayers = 3  #default 3,
    _batch_size = 10000 #default 100
    _init_size = 30.000 #3*batch_size
    _n_init = 20 #default 10
    _reassignment_ratio =  0.1 #default 0.01
    if SIFT_TYPE == "sift":
        _contrastThreshold = 0.01   #default 0.04,
        _edgeThreshold = 13 #default 10
        _sigma = 0.95 #default 1.6 
        K_MEANS_K = 128
    elif SIFT_TYPE=="dense_sift":
        _contrastThreshold = 0#0.03 #default 0.04,
        _edgeThreshold = 0#11   #default 10
        _sigma = 1.4    #default 1.6 
        DENSE_STEP_SIZE = 15 #default 10
        K_MEANS_K = 256
    else:
        print("unrecognized sift type")
        exit()
    
    global to_write
    to_write["nfeatures"] = _nfeatures
    to_write["OctaveLayers"] = _nOctaveLayers
    to_write["contrastThreshold"] = _contrastThreshold
    to_write["edgeThreshold"] = _edgeThreshold
    to_write["sigma"] = _sigma
    to_write["init_size"] = _init_size
    to_write["n_init"] = _n_init
    to_write["reassignment_ratio"] = _reassignment_ratio
    to_write["batch_size"] = _batch_size
    to_write["step-size"] = DENSE_STEP_SIZE

def randomize_parameters():
    global _nfeatures
    _nfeatures = generate_random(500,10000,0.7,0,intmi=True)    #default 0
    global _nOctaveLayers
    _nOctaveLayers = generate_random(1,7,0.7,3,intmi=True)  #default 3,
    global _contrastThreshold
    _contrastThreshold = generate_random(0.01,0.1,0.7,0.04) #default 0.04,
    global _edgeThreshold
    _edgeThreshold = generate_random(3,20,0.7,10)   #default 10
    global _sigma
    _sigma = generate_random(0.1,4) #default 1.6 
    global _batch_size
    _batch_size = generate_random(100,500,0.3,100,intmi=True) #default 100
    global _init_size
    _init_size = 3*_batch_size #3*batch_size
    global _n_init
    _n_init = 10 #default 10
    global _reassignment_ratio
    _reassignment_ratio = generate_random(0.005,0.05) #default 0.01
    global DENSE_STEP_SIZE
    DENSE_STEP_SIZE = generate_random(3,20,0.2,10,intmi=True) #default 10
    global K_MEANS_K
    K_MEANS_K = choice([16,32,64,128,256])
    global KNN_K
    KNN_K = choice([2,4,8,16,32,64])
    global SIFT_TYPE
    SIFT_TYPE = choice(["sift","dense_sift"])

    global to_write
    to_write["nfeatures"] = _nfeatures
    to_write["OctaveLayers"] = _nOctaveLayers
    to_write["contrastThreshold"] = _contrastThreshold
    to_write["edgeThreshold"] = _edgeThreshold
    to_write["sigma"] = _sigma
    to_write["init_size"] = _init_size
    to_write["n_init"] = _n_init
    to_write["reassignment_ratio"] = _reassignment_ratio
    to_write["batch_size"] = _batch_size
    to_write["step-size"] = DENSE_STEP_SIZE
    to_write["kmeans_k"] = K_MEANS_K
    to_write["sift_type"] = SIFT_TYPE
    to_write["knn_k"] = KNN_K

def log_results(save_to="k",log=True):
    
    global to_write

    with open("result_log.txt","a") as filem:
        filem.write("\nParams : %s \n" % str(to_write) )
        filem.write("-------------" )
    with open("result_log.p","wb") as filem:
        pc.dump(to_write,filem)




## This series of methods' returns were definite at some point
## Now their return value further edited at couple of places
## But usually is not
def get_feat_filename():
    if SIFT_TYPE=="sift":
            feat_filename = "features.p"
    elif SIFT_TYPE == "dense_sift":
        feat_filename = "dense_features.p"
    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()
    return feat_filename

def get_result_filename():
    if SIFT_TYPE=="sift":
            feat_filename = "result_"+str(K_MEANS_K)+"_"+str(KNN_K)+".txt"
    elif SIFT_TYPE == "dense_sift":
        feat_filename = "dense_result_"+str(K_MEANS_K)+"_"+str(KNN_K)+".txt"
    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()
    return feat_filename


def get_bof_filename():
    if SIFT_TYPE=="sift":
            feat_filename = "bof_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"
    elif SIFT_TYPE == "dense_sift":
        feat_filename = "dense_bof_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"

    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()
    return feat_filename

def get_centers_filename():
    if SIFT_TYPE=="sift":
            feat_filename = "centers_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"
    elif SIFT_TYPE == "dense_sift":
        feat_filename = "dense_centers_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"
    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()
    return feat_filename

def get_kmeans_filename():
    if SIFT_TYPE=="sift":
            feat_filename = "kmeans_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"
    elif SIFT_TYPE == "dense_sift":
        feat_filename = "dense_kmeans_"+str(K_MEANS_K)+"_"+str(KNN_K)+".p"
    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()
    return feat_filename

def get_image_names():
    retval = glob.glob("./the2_data/train/*/*")
    return retval

def get_val_image_names():
    retval = glob.glob("./the2_data/validation/*/*")
    return retval

def get_classes():
    return ["apple","beetle","crab","elephant","lion","orange","road","woman","aquarium_fish","camel","cup","flatfist","mushroom","pear","skyscraper"]

def eucledian(v1,v2):
    return np.linalg.norm(v1-v2)


def extract_features_of_image(img,sift = None):
    
    if(sift is None):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures = _nfeatures, nOctaveLayers=_nOctaveLayers,\
                            contrastThreshold=_contrastThreshold,edgeThreshold=_edgeThreshold,\
                            sigma=_sigma)

    if SIFT_TYPE=="sift":
        kp = sift.detect(img,None)
        if(len(kp) < 5 ):
            print("Found an image with %d keypoints" % len(kp))
            kp = [cv2.KeyPoint(x, y, 100) for y in range(0, img.shape[0], 100) 
                                                for x in range(0, img.shape[1], 100)]
    elif SIFT_TYPE=="dense_sift":

        kp = [cv2.KeyPoint(x, y, DENSE_STEP_SIZE) for y in range(0, img.shape[0], DENSE_STEP_SIZE) 
                                            for x in range(0, img.shape[1], DENSE_STEP_SIZE)]
        
    else:
        print("UNRECOGNIZED SIFT TYPE, aborting")
        exit()


    vector = sift.compute(img,kp)

    vector = vector[-1]
    
    return vector

def extract_all_feature_vectors():
    pic_names_array = get_image_names()
    
    feat_filename = get_feat_filename()
    number_of_features = 0

    with open(feat_filename,"wb") as feat_file:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures = _nfeatures, nOctaveLayers=_nOctaveLayers,\
                            contrastThreshold=_contrastThreshold,edgeThreshold=_edgeThreshold,\
                            sigma=_sigma)
        for index,pic in enumerate(pic_names_array):
            #if(index%100==0):
            #   print("Feature extracted %d images" % index)
            curr_image_gray = cv2.imread(pic,0) #0=read image as gray
            features = extract_features_of_image(curr_image_gray,sift)
            number_of_features += features.shape[0]
            #features = list(features)
            #map(lambda x:list(x),features)
            pc.dump(features,feat_file)

    return number_of_features, pic_names_array


def load_all_features(num_feat,image_names=""):
    feat_filename = get_feat_filename()
    number_of_features = num_feat

    all_features = np.zeros(shape=(number_of_features,128))

    featur_count_per_image = []

    with open(feat_filename,"rb") as feat_file:
        index = 0
        while True:
            #if(index%1000==0):
            #   print("Loaded %d features" % index)
            try:
                current_pic_features = pc.load(feat_file)
            except EOFError:
                break
            featur_count_per_image.append(current_pic_features.shape[0])
            for feat in current_pic_features:
                all_features[index] = feat
                index+=1

    assert len(featur_count_per_image)==len(image_names)==NUMBER_OF_IMAGES,"Different counts"
    return all_features,featur_count_per_image



def kmeans_and_save_clusters_and_save_bofs(all_features,names,counts):
    """
    _init_size = 300 #3*batch_size
    _n_init = 10 #default 10
    _reassignment_ratio = 0.01 #default 0.01
    _batch_size = 100 #default 100
    """

    kmeans = cluster.MiniBatchKMeans(n_clusters=K_MEANS_K,verbose=0,n_init=_n_init,\
                compute_labels=True,reassignment_ratio=_reassignment_ratio,\
                batch_size=_batch_size)
    kmeans.fit(all_features)
    print("Done k means")
    centers_filename = get_centers_filename()
    kmeans_filename = get_kmeans_filename()
    labels = kmeans.labels_

    with open(centers_filename,"wb") as filem:
        pc.dump(kmeans.cluster_centers_,filem)

    with open(kmeans_filename,"wb") as filem:
        pc.dump(kmeans,filem)

    pic_names_array = get_image_names()
    bof_filename = get_bof_filename()
    number_of_clusters = kmeans.cluster_centers_.shape[0]
    assert number_of_clusters==K_MEANS_K,"Sanity check"
    all_bof_vectors = np.zeros((NUMBER_OF_IMAGES,number_of_clusters))
    curr_count = 0
    for img_index in range(len(counts)):
        #if(img_index%10==0):
        #   print("Bof featured %d image" %img_index)
        
        temp = labels[curr_count:curr_count+counts[img_index]].astype(np.float32)
        bof_vector = np.histogram(temp, bins=np.arange(number_of_clusters+1),density=True)[0]
        curr_count+=counts[img_index]
        ####slice the image name
        img_name = names[img_index]
        img_name = img_name[img_name.rfind("/")+1:]
        ###

        all_bof_vectors[img_index] = bof_vector 
        
    with open(bof_filename,"wb") as filem:
        pc.dump(all_bof_vectors,filem)


    return kmeans.cluster_centers_



def extract_bof(img_name,all_bof_vectors = None):

    if all_bof_vectors is None:
        bof_filename = get_bof_filename()

        with open(bof_filename,"rb") as filem:
            all_bof_vectors = pc.load(filem)
        
   
    kmeans_filename = get_kmeans_filename()
    bof_filename = get_bof_filename()

    curr_image_gray = cv2.imread(img_name,0) #0=read image as gray
    features = extract_features(curr_image_gray)

    with open(kmeans_filename,"rb") as filem:
        kmeans = pc.load(filem)
        if(features.shape[0]!=0):
            current_bof_vector = kmeans.predict(features).astype(np.float32)
            current_bof_vector = np.histogram(current_bof_vector, bins=np.arange(K_MEANS_K+1),density=True)[0]
            #normalize
            current_bof_vector /= features.shape[0]
        else:
            print("PROBLEM")
            current_bof_vector = np.zeros((kmeans.cluster_centers_.shape[0],))

    return current_bof_vector


def predict(val_bofs,all_bofs,classes):
       knn_ = KNN(KNN_K,n_jobs=-1)
       knn_.fit(all_bofs,classes)
       return knn_.predict(val_bofs)
    

def class_from_path_train(p):
    p = p[18:]
    t = p.rfind("/")
    p = p[:t]
    return p
def class_from_path_val(p):
    p = p[23:]
    t = p.rfind("/")
    p = p[:t]
    return p

def query_validation_set():
    validation_paths = get_val_image_names()
    result_filename = "val_"+get_result_filename()
    bof_filename = get_bof_filename()
    true_preds = 0
    with open(bof_filename,"rb") as bof_file:
        all_bof_vectors = pc.load(bof_file)

    ##calc classes
    names = get_image_names()
    classes = []
    for n in names:
        classes.append( class_from_path_train(n))
    ##
    val_bof_vectors = np.zeros((len(validation_paths),all_bof_vectors.shape[1]))
    for index, line in enumerate(validation_paths):
        curr_val_image_path = line.rstrip('\n')
        val_bof_vectors[index] = extract_bof(curr_val_image_path,all_bof_vectors)
        
    val_preds = predict(val_bof_vectors,all_bof_vectors,classes)
    with open(result_filename,"w") as res_file:
            for index, line in enumerate(validation_paths):
                curr_val_image_path = line.rstrip('\n')
                gt_label = class_from_path_val(curr_val_image_path)
                if gt_label == val_preds[index]:
                    true_preds += 1
                res_file.write(curr_val_image_path+": "+val_preds[index]+'\n')

    print("Accuracy: %f" % (true_preds*100/val_preds.shape[0]) )
    global to_write
    to_write["val_score"] = true_preds*100/val_preds.shape[0]

    
def full_pipeline(k=-1):

    global K_MEANS_K
    global to_write

    if k>0:
        K_MEANS_K = k
    print("K= ",K_MEANS_K)
    image_names = get_image_names()
    
    total_feature_count,image_names = extract_all_feature_vectors()

    if SIFT_TYPE=="sift":
        to_write["total_feature_count-sift"] = total_feature_count
    elif SIFT_TYPE=="dense_sift":
        to_write["total_feature_count-dense"] = total_feature_count

    print("Total feature : ",total_feature_count)

    
    all_features, feat_counts = load_all_features(total_feature_count,image_names)
    print("All feature shape : ",all_features.shape)
    
    clusters = kmeans_and_save_clusters_and_save_bofs(all_features, image_names,feat_counts)
    print("Saved bof_pairs, Cluster shape : ",clusters.shape)
    
    query_validation_set()
    print("Queried and saved validation set")

    print()


def iterate_all_k_values():
    full_pipeline(32)
    full_pipeline(64)
    full_pipeline(128)
    full_pipeline(256)
    full_pipeline(512)

def go_full_blown_crazy():
    global to_write
    while True:
        try:
            randomize_parameters()
            print()
            print("Working for: ")
            print(str(to_write))
            print()
            full_pipeline()
            print("Finished, logging")
            """
            print("Finished for normal sift, starting for dense")
            SIFT_TYPE = "dense_sift"
            full_pipeline(256)
            print("Finished dense too,logging")
            """
            log_results()
            print()
            print("--------------------")
        except MemoryError:
            print("Memory Error, continuing:")
            print()
            print()
            continue

def search_k():
    global to_write
    global SIFT_TYPE
    global _sigma
    SIFT_TYPE = "sift"
    for k in [80,90,100,110,140,150,160,170,200]:
        try:
            
            print("Working for k: %d" % k)
            print(str(to_write))
            print()
            full_pipeline(k)
            print("Finished, logging")
            log_results(k)
            print()
            print()
        except MemoryError:
            print("Memory Error, continuing:")
            print()
            print()
            continue




if __name__ == '__main__':
    go_full_blown_crazy()
    #search_ct()
    #go_full_blown_crazy()
    #query_validation_set()
    #all_features = update_feature_vectors()
    #total_features = load_all_features()
