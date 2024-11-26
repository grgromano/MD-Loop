class CUSUM():
  # Cumulative Sum Control Chart changepoint detection algorithm

  def __init__(self, N, eps, threshold):

    self.N = N # number of samples needed to initialize the reference value
    self.reference = 0 # reference value
    self.eps = eps # epsilon value    
    self.threshold = threshold # threshold
    self.g_plus = 0 # g values
    self.g_minus = 0
    self.t = 0 # number of rounds executed


  def update(self, sample):
    self.t += 1 

    if self.t <= self.N:
      self.reference += sample/self.N
      #print("REFERENCE: ", self.reference, "Sample:", sample)
      return False
    
    else:
      self.reference = (self.reference*(self.t-1) + sample)/self.t
      s_plus = (sample - self.reference) - self.eps
      s_minus = -(sample - self.reference) - self.eps
      self.g_plus = max(0, self.g_plus + s_plus)
      self.g_minus = max(0, self.g_minus + s_minus)
      # print("REFERENCE: ", self.reference)
      # print("s_plus: ", s_plus, "g_plus: ", self.g_plus)
      # print('')
      
      if self.g_plus > self.threshold or self.g_minus > self.threshold:
        self.reset()
        print('Change-point detected!')
        return True
      return False
    

  def reset(self):
    self.t = 0
    self.g_plus = 0
    self.g_minus = 0
    self.reference = 0