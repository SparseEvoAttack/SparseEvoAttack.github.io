import numpy as np

from utils_pgd import get_predictions, get_predictions_and_gradients, get_predictions_norm, get_predictions_and_gradients_norm,get_predictions_and_gradients_norm_target,get_predictions_norm_target

def project_L0_box(y, k, lb, ub):
  ''' projection of the batch y to a batch x such that:
        - each image of the batch x has at most k pixels with non-zero channels
        - lb <= x <= ub '''
      
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    
  return x
   
def perturb_L0_box_norm(attack, x_nat, y_nat, lb, ub,mu,std):
  ''' PGD attack wrt L0-norm + box constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - lb <= adv - x_nat <= ub
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = x_nat.copy()

  for i in range(attack.num_steps):
    if i > 0:
      pred, grad = get_predictions_and_gradients_norm(attack.model, x2, y_nat,mu,std)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
    
    x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)
    if len(adv_not_found)==1 and (adv_not_found==0):
        break
        
  return adv, adv_not_found

def perturb_L0_box_norm_target(attack, x_nat, y_nat, lb, ub,mu,std):
  ''' PGD attack wrt L0-norm + box constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - lb <= adv - x_nat <= ub
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = x_nat.copy()

  for i in range(attack.num_steps):
    if i > 0:
      pred, grad = get_predictions_and_gradients_norm_target(attack.model, x2, y_nat,mu,std)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)
    if len(adv_not_found)==1 and (adv_not_found==0):
        break
    
  return adv, adv_not_found

class PGDattack():
  def __init__(self, model, args):
    self.model = model
    self.type_attack = args['type_attack'] # 'L0', 'L0+Linf'
    self.num_steps = args['num_steps']     # number of iterations of gradient descent for each restart
    self.step_size = args['step_size']     # step size for gradient descent (\eta in the paper)
    self.n_restarts = args['n_restarts']   # number of random restarts to perform
    self.rs = True                         # random starting point
    self.epsilon = args['epsilon']         # for L0+Linf, the bound on the Linf-norm of the perturbation
    self.k = args['sparsity']              # maximum number of pixels that can be modified (k_max in the paper)
    self.mu = args['mu']
    self.std = args['std']

  def perturb_norm(self, x_nat, y_nat,arch=None):
    adv = np.copy(x_nat)
    if self.type_attack == 'L0+sigma': self.sigma = sigma_map(x_nat)
      
    for counter in range(self.n_restarts):
      if counter == 0:
        corr_pred = get_predictions_norm(self.model, x_nat, y_nat,self.mu,self.std)
        pgd_adv_acc = np.copy(corr_pred)
        
      if self.type_attack == 'L0':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box_norm(self, x_nat, y_nat, -x_nat, 1.0 - x_nat,self.mu,self.std)
      
      elif self.type_attack == 'L0+Linf':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box_norm(self, x_nat, y_nat, np.maximum(-self.epsilon, -x_nat), np.minimum(self.epsilon, 1.0 - x_nat))
      
      pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
      adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
    
    pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1,2))
    corr_pred = get_predictions_norm(self.model, adv, y_nat,self.mu,self.std)
    
    return adv, pgd_adv_acc,pixels_changed

  def perturb_norm_target(self, x_nat, y_nat,arch=None):
    adv = np.copy(x_nat)
    if self.type_attack == 'L0+sigma': self.sigma = sigma_map(x_nat)
      
    for counter in range(self.n_restarts):
      if counter == 0:
        corr_pred = get_predictions_norm_target(self.model, x_nat, y_nat,self.mu,self.std)
        pgd_adv_acc = np.copy(corr_pred)
        
      if self.type_attack == 'L0':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box_norm_target(self, x_nat, y_nat, -x_nat, 1.0 - x_nat,self.mu,self.std)
      
      elif self.type_attack == 'L0+Linf':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box_norm_target(self, x_nat, y_nat, np.maximum(-self.epsilon, -x_nat), np.minimum(self.epsilon, 1.0 - x_nat))
      
      pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
      adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
    
    pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1,2))
    corr_pred = get_predictions_norm_target(self.model, adv, y_nat,self.mu,self.std)
    
    return adv, pgd_adv_acc,pixels_changed