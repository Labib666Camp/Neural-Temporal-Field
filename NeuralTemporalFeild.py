from sklearn.neural_network import MLPRegressor
import numpy as np

class NeuralTemporalField:
  def __init__(self,signal,solver,activation,max_itr,hidden_layers):
    self.signal = signal
    self.solver = solver
    self.activation = activation
    self.itr = max_itr
    self.hidden_layers = hidden_layers

  def Train_Test(self):
    sh = self.signal.shape[0] 
    s_en = int (sh*(90/100))
    x_train = self.signal[0:s_en,:]
    y_train = self.signal[1:(s_en+1),:]
    x_test = self.signal[(s_en+1):-1,:]
    y_test = self.signal[(s_en+2):,:]
    return x_train,y_train,x_test,y_test

  def Evaluate(self):
    # here we only work with dimension x0
    dxs,k = self.Perturbations()
    no_dims = self.signal.shape[1]
    #template with all mins
    template = np.array([np.amin(self.signal[:,d]) for d in range(no_dims)])
    samples = []
    for p in range(100):
      if p==k:
        samples.append(self.signal[0,:])
      else:
        temp = template+dxs
        samples.append(temp)
        template = temp
    return np.array(samples)

  def Perturbations(self):
    dxes = []
    no_dims = self.signal.shape[1]
    for d in range(no_dims):
      xd = self.signal[:,d]
      mx = np.amax(xd)
      mn = np.amin(xd)
      step = (mx - mn)/100
      dxes.append(step)
      if d==0:
        k = int ((xd[0] - mn)/step)
    return np.array(dxes),k

  def fit_transform(self):
    x_train,y_train,x_test,y_test = vector_train_test(self.signal)
    regr = MLPRegressor(random_state=1, max_iter=self.itr,
                        solver=self.solver,activation=self.activation,
                        hidden_layer_sizes=self.hidden_layers).fit(x_train, y_train)
    score = regr.score(x_test, y_test)
    r = []
    x_eval = self.Evaluate()
    r.append(x_eval)
    for t in range(1,100):
      t_next = regr.predict(x_eval)
      r.append(t_next)
      x_eval = t_next
    return np.array(r)
