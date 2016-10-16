import numpy
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_train = [ [12, 3, 4, 5], [16, 4, 3, 8]]
y_train = [ 16, 20]
X_test = [[28, 7, 5, 10]]


#Generate more training examples
X_generated=[]
y_generated=[]
for i in range(3000):
  multiplier = randint(1,10)
  a = randint(1,5)
  b=a*multiplier
  c=randint(1,5)
  d=randint(1,5)
  e = c*d - multiplier
  X_generated.append([a, b, c, d])
  y_generated.append(e)


#print X_generated
#print y_generated
X_train.extend(X_generated)
y_train.extend(y_generated)
#print X_train
#print y_train

'''
for i in range(8):
  X_train.extend(X_train)
  y_train.extend(y_train)
'''
print len(X_train), len(y_train)
def  baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(120, input_dim=4, init='normal', activation='relu'))
  model.add(Dense(1, init='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
estimator.fit(X_train, y_train, nb_epoch=1500, batch_size=100)
X_test = numpy.array(X_test)
y_predicted = estimator.predict(X_test)
print y_predicted
