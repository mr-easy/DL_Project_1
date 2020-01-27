import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

# Uncomment the following lines if tensorflow gives error.
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
###################### Till here
#print(sys.argv)

inputFileName = "test_input.txt"
flag = "test"
if(len(sys.argv) > 1 and sys.argv[1] == "--test-data"):
    flag = "test"
elif(len(sys.argv) > 1 and sys.argv[1] == "--train"):
    flag = "train"
if(len(sys.argv) > 2):
    inputFileName = str(sys.argv[2])

bits = 16  # number of bits used to represent the input number

print("IISc Deartment: CSA")
print("Person Name: Rishabh Gupta")

def model_creation():
    model = keras.Sequential([
        #keras.layers.Reshape(target_shape=(bits,), input_shape=(bits, )),
        keras.layers.Dense(units=1000, activation='relu', input_shape=(bits,)),
        #keras.layers.Dropout(0.3),
        #keras.layers.Dense(units=512, activation='relu'),
        #keras.layers.Dropout(0.3),
        #keras.layers.Dense(units=256, activation='relu'),
        #keras.layers.Dropout(0.3),
        #keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


if(flag == "test"):

    # Read input file
    testNumbers = [int(number.rstrip('\n')) for number in open(inputFileName)]
    
    # Software 1 approach
    results_1 = ["fizzbuzz" if v%15 == 0 else "fizz" if v%3 == 0 else "buzz" if v%5 == 0 else str(v) for v in testNumbers]
    #print(results_1)
    with open("Software1.txt", "w") as f:
        f.write("\n".join(results_1))
        print("Software1.txt generated.")
        
    # Software 2 approach
    
    # load model
    model = model_creation()
    model.load_weights('./model/model_weights')
    #model = keras.models.load_model('./model')
    
    # To predictions
    predictions = model.predict([[int(x) for x in f'{i:0{bits}b}'] for i in testNumbers])
    predictedClass = np.array([np.argmax(pred) for pred in predictions])
    results_2 = []
    for i in range(len(testNumbers)):
        results_2.append(["fizzbuzz", "fizz", "buzz", str(testNumbers[i])][predictedClass[i]])        
    
    with open("Software2.txt", "w") as f:
        f.write("\n".join(results_2))
        print("Software2.txt generated.")

    #correct = 0
    #for i in range(len(results_1)):
    #    if(results_1[i] == results_2[i]):
    #        correct += 1   
    #                                                 
    #print("Accuracy = " + str(correct/len(testNumbers)))           
                                                                                             
else:  #train
    
    print("Training...")
    # Prepare dataset
    xtrain = np.array([[int(x) for x in f'{i:0{bits}b}'] for i in range(101, 1001)])
    ytrain = np.array([0 if v%15 == 0 else 1 if v%3 == 0 else 2 if v%5 == 0 else 3 for v in range(101, 1001)])
    
    def preprocess(x, y):
        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int64)
        return x, y

    def create_dataset(xs, ys, n_classes=4):
        ys = tf.one_hot(ys, depth=n_classes)
        return tf.data.Dataset.from_tensor_slices((xs, ys)) \
            .map(preprocess) \
            .shuffle(len(ys)) \
            .batch(128)
                                           
    trainingData = create_dataset(xtrain, ytrain)
    
    # Define model                                       
    model = model_creation()                                        
    print(model.summary())
    
    # Train model                                       
    history = model.fit( 
        trainingData,  
        epochs=5000,
        verbose=1
    )             
                                           
    # Save model 
    model.save_weights('./model/model_weights')
    print("Model saved.")