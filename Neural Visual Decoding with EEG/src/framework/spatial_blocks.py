import torch
import torch.nn as nn

import math
from src.framework.hyperparameters import *

class SpatialFeatureExtraction_Module():
    def __init__(self, R=100, S=50):
      self.recording_splitter = RecordingSplitter(R)
      self.scm_computer = SCMsComputer()
      self.dim_red_operator = DimensionalityReductionOperator(S)
      self.tan_sp_learning_operator = TangentSpaceLearningOperator()
      self.output_space_processor = Vectorizer()

    def forward(self, x):
      eeg_splitted = self.recording_splitter.forward(x)
      if torch.isnan(eeg_splitted).any():
            print("NaN values detected in eeg_splitted")
      scms, subband_average_covariances = self.scm_computer.forward(eeg_splitted)
      if torch.isnan(scms).any():
            print("NaN values detected in scms")
      scms_reduced, subband_average_covariances_reduced = self.dim_red_operator.forward(scms, subband_average_covariances)
      if torch.isnan(scms_reduced).any():
            print("NaN values detected in scms_reduced")
      S = self.tan_sp_learning_operator.forward(scms_reduced, subband_average_covariances_reduced)
      if torch.isnan(S).any():
            print("NaN values detected in S")
      space_features = self.output_space_processor.forward(S)
      if torch.isnan(space_features).any():
            print("NaN values detected in space_features")
      return space_features


class SpatialFeatureProcessing_Module(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2, dropout_rate=0.5):
        """
        input_size = H*P*S(S+1)/2
        hidden_size_fc1 = 512
        hidden_size_fc2 = 64
        """
        super(SpatialFeatureProcessing_Module, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)

        self.dropout = nn.Dropout(dropout_rate)        

    def forward(self, x):
        """
        input tensor x of size [H,P,S(S+1)/2]
        """
        x = x.to(device)
        x = x.float()
        x = x.reshape(x.size(0), -1) # size [input_size=H*P*S(S+1)/2] # Recall we pass data in batch 
        # Fully connected Layer
        x = self.fc1(x) # size [hidden_size_fc1=512]
        x = self.dropout(x)

        # Fully connected Layer
        x = self.fc2(x) # size [hidden_size_fc2=64]
        x = self.dropout(x)

        return x


class RecordingSplitter():
    def __init__(self, R):
      super().__init__()
      self.R = R # number of time samples within the segment, s.t. P = T//R

    def forward(self, x):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]

      batch_output = []
      for batch in range(B):

          output = []
          for subband in range(H):

            segments = torch.split(x[batch][subband], self.R, dim=1)
            tensor_segments = torch.stack(segments, dim=0)
            output.append(tensor_segments)

          output = torch.stack(output, dim=0)
          batch_output.append(output)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output


class SCMsComputer():
    def __init__(self):
      super().__init__()

    def forward(self, x):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      P = x.shape[2]

      batch_output = []
      batch_subband_average_covariances = []
      for batch in range(B):

          output = []
          subband_average_covariances = []
          for subband in range(H):

            recording_scms = []
            for segment in range(P):

              scm_matrix = self.scm(x[batch][subband][segment])
              recording_scms.append(scm_matrix)

            average_covariance = sum(recording_scms) / len(recording_scms)
            recording_scms = torch.stack(recording_scms, dim=0)
            output.append(recording_scms)
            subband_average_covariances.append(average_covariance)

          output = torch.stack(output, dim=0)
          batch_output.append(output)
          batch_subband_average_covariances.append(subband_average_covariances)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output, batch_subband_average_covariances

    def scm(self, matrix):
      # x: [N, R]
      R = matrix.shape[1]
      return torch.mm(matrix, matrix.t()) / (R - 1)


class DimensionalityReductionOperator():
    def __init__(self, S):
      super().__init__()
      self.S = S # top S eigenvalues of the average covariance matrix      

    def forward(self, x, subband_average_covariances):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      P = x.shape[2]

      batch_output = []
      batch_subband_average_covariances_reduced = []
      for batch in range(B):

          output = []
          subband_average_covariances_reduced = []
          for subband in range(H):
            self.average_covariance = subband_average_covariances[batch][subband]
            self.W = self.pca(self.average_covariance)[:, :self.S]

            recording_reduced_scm = []
            for segment in range(P):

              C = x[batch][subband][segment]
              reduced_scm = self.dot_product(self.W.t(), C, self.W) # size [S,S]
              recording_reduced_scm.append(reduced_scm)

            average_covariance_reduced = sum(recording_reduced_scm) / len(recording_reduced_scm)
            recording_reduced_scm = torch.stack(recording_reduced_scm, dim=0)
            output.append(recording_reduced_scm)
            subband_average_covariances_reduced.append(average_covariance_reduced)

          output = torch.stack(output, dim=0)
          batch_output.append(output)
          batch_subband_average_covariances_reduced.append(subband_average_covariances_reduced)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output, batch_subband_average_covariances_reduced

    def pca(self, x):
      # x: [N, N]
      eigenvalues, eigenvectors = torch.linalg.eigh(x, UPLO='U') #torch.symeig(x, eigenvectors=True)
      indices = torch.argsort(eigenvalues, descending=True)
      return eigenvectors[:, indices]

    def dot_product(self, Wt, C, W):
      # W: [N, S]
      # C: [N, N]
      intermediate = torch.mm(Wt, C)
      return torch.mm(intermediate, W)


class TangentSpaceLearningOperator():
    def __init__(self, epsilon=10**(-9)):
      super().__init__()
      self.epsilon = epsilon     

    def forward(self, x, subband_average_covariances_reduced):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      P = x.shape[2]

      batch_output = []
      for batch in range(B):    

          output = []
          for subband in range(H):
            self.Cref = self.riemann_mean_estimation(subband_average_covariances_reduced[batch][subband], x[batch][subband], self.epsilon)
            if torch.isnan(self.Cref).any():
              print("NaN values detected in self.Cref")
            self.Cref_minus05 = self.power(self.Cref, -0.5)
            if torch.isnan(self.Cref_minus05).any():
              print("NaN values detected in self.Cref_minus05")

            recording_S = []
            for segment in range(P):

              C = x[batch][subband][segment]
              S = self.logarithm(torch.mm(self.Cref_minus05, torch.mm(C, self.Cref_minus05))) # size [S,S]
              S = torch.sqrt(torch.real(S)**2 + torch.imag(S)**2)
              recording_S.append(S)

            recording_S = torch.stack(recording_S, dim=0)
            output.append(recording_S)

          output = torch.stack(output, dim=0)
          batch_output.append(output)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output

    def riemann_mean_estimation(self, average_covariance_reduced, x_subband, epsilon):
      # Initialize Cref
      Cref = average_covariance_reduced
      # Initialize J
      log_mapping_Cref = []
      for C in x_subband:
        log_mapping_Cref.append(self.logarithm_mapping_C(C, Cref))
      J = sum(log_mapping_Cref) / len(log_mapping_Cref)
      # Loop based on the condition on the frobenius norm of J
      J_frobenius_norm = torch.norm(J, p=2)
      while J_frobenius_norm < epsilon:
        # update Cref
        Cref = self.exponential_mapping_C(J, Cref)
        # update J
        log_mapping_Cref = [self.logarithm_mapping_C(C, Cref) for C in x_subband]
        J = sum(log_mapping_Cref) / len(log_mapping_Cref)

      return Cref
    
    def power(self, M, p):
      # M: [S, S]
      L = torch.linalg.cholesky(M, upper=False)
      sqrt = torch.mm(L, L.t())
      sqrt_inv = torch.inverse(sqrt)
      result = torch.pow(sqrt_inv, p)
      #Replace NaNs with a small value
      result[torch.isnan(result)] = self.epsilon
      return result

    def logarithm(self, M):
      # M: [S, S]
      # Replace infs with a small value
      M[torch.isinf(M)] = self.epsilon
      # Replace NaNs with a small value
      M[torch.isnan(M)] = self.epsilon
      eigenvalues, eigenvectors = torch.linalg.eig(M)
      log_eigenvalues = torch.log(eigenvalues[0]).expand_as(eigenvalues)
      log_M = torch.mm(eigenvectors, torch.mm(torch.diag(log_eigenvalues), eigenvectors.t()))
      # Replace NaNs with a small value
      log_M[torch.isnan(log_M)] = self.epsilon
      return log_M

    def exponential(self, M):
      # M: [S, S]
      # Replace infs with a small value
      M[torch.isinf(M)] = self.epsilon
      # Replace NaNs with a small value
      M[torch.isnan(M)] = self.epsilon
      eigenvalues, eigenvectors = torch.linalg.eig(M)
      exp_eigenvalues = torch.exp(eigenvalues[0]).expand_as(eigenvalues)
      exp_M = torch.mm(eigenvectors, torch.mm(torch.diag(exp_eigenvalues), eigenvectors.t()))
      return exp_M

    def logarithm_mapping_C(self, Cprime, C):
      C_plus05 = self.power(C, 0.5)
      C_minus05 = self.power(C, -0.5)
      log = self.logarithm(torch.mm(C_minus05, torch.mm(Cprime,C_minus05)))
      log_mapping = torch.mm(C_plus05.to(torch.complex128), torch.mm(log, C_plus05.to(torch.complex128)))
      return log_mapping

    def exponential_mapping_C(self, Cprime, C):
      C_plus05 = self.power(C, 0.5)
      C_minus05 = self.power(C, -0.5)
      exp = self.exponential(torch.mm(C_minus05, torch.mm(Cprime,C_minus05)))
      exp_mapping = torch.mm(C_plus05, torch.mm(exp, C_plus05))
      return exp_mapping


class Vectorizer():
    def __init__(self):
      super().__init__()

    def forward(self, x):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      P = x.shape[2]

      batch_output = []
      for batch in range(B):

          output = []
          for subband in range(H):

            recording_upper_S = []
            for segment in range(P):

              S_matrix = x[batch][subband][segment]
              upper_S = self.half_vectorize(S_matrix)
              recording_upper_S.append(upper_S)

            recording_upper_S = torch.stack(recording_upper_S, dim=0)
            output.append(recording_upper_S)

          output = torch.stack(output, dim=0)
          batch_output.append(output)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output

    def half_vectorize(self, M):
      S = M.shape[0]
      half_vectorized = torch.zeros(S*(S+1)//2)
      idx = 0
      for i in range(S):
          for j in range(i, S):
              if i == j:
                  half_vectorized[idx] = M[i, j]
              else:
                  half_vectorized[idx] = M[i, j] * math.sqrt(2)
              idx += 1
      return half_vectorized

