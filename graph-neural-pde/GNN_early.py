"""
A GNN used at test time that supports early stopping during the integrator
"""

import torch
import torch.nn.functional as F
import argparse
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
from data import get_dataset
# from run_GNN import get_optimizer, train, test
from early_stop_solver import EarlyStopInt
from base_classes import BaseGNN
from model_configurations import set_block, set_function

from adjoint import odeint, odeint_adjoint
from hyperbolic_geometry import PoincareBall

class GNNEarly(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNNEarly, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    self.device = device
    time_tensor = torch.tensor([0, self.T]).to(device)
    # self.regularization_fns = ()
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
    # overwrite the test integrator with this custom one
    with torch.no_grad():
      self.odeblock.test_integrator = EarlyStopInt(self.T, self.opt, self.device)
      self.set_solver_data(dataset.data)

    self.c = 1.0  # Curvature of Poincaré ball
    self.manifold = PoincareBall()  # Trustworthy Poincaré ball implementation
    # self.odeblock = ODEBlock(self.odefunc, self.time_range, args)



  def set_solver_m2(self):
    self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
    self.odeblock.test_integrator.m2_bias = self.m2.bias.data.detach().clone().to(self.device)

  def set_solver_data(self, data):
    self.odeblock.test_integrator.data = data


  def cleanup(self):
    del self.odeblock.test_integrator.m2
    torch.cuda.empty_cache()


  def forward(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    # self.odeblock.set_x0(x)

    # Project to hyperbolic space if using hyperbolic geometry
    if self.opt['function'] == 'hyperbolic_laplacian':
      # x = torch.clamp(x, min=-5.0, max=5.0)
      x = self.manifold.proj_tan0(x, self.c)
      # x = 0.5 * x
      x = self.manifold.expmap0(x, c=self.c)
      x = self.manifold.proj(x, c=self.c)

    self.odeblock.set_x0(x)



    with torch.no_grad():
      self.set_solver_m2()

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    # Project back to tangent space if using hyperbolic geometry
    if self.opt['function'] == 'hyperbolic_laplacian':
      z = self.manifold.proj(z, c=self.c)
      z = self.manifold.logmap0(z, self.c)
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

  def forward_encoder(self, x, pos_encoding):
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = self.mx(x)
      p = self.mp(pos_encoding)
      x = torch.cat([x, p], dim=1)
    else:
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = x + self.m11(F.relu(x))
      x = x + self.m12(F.relu(x))

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def forward_ODE(self, x, pos_encoding):
    x = self.forward_encoder(x, pos_encoding)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    return z


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat, pos_encoding), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


def main(opt):
  dataset = get_dataset(opt, '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, data = GNNEarly(opt, dataset, device).to(device), dataset.data.to(device)
  print(opt)
  # todo for some reason the submodule parameters inside the attention module don't show up when running on GPU.
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_val_acc = test_acc = best_epoch = 0
  best_val_acc_int = best_test_acc_int = best_epoch_int = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()
    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    val_acc_int = model.odeblock.test_integrator.solver.best_val
    tmp_test_acc_int = model.odeblock.test_integrator.solver.best_test
    # store best stuff inside integrator forward pass·
    if val_acc_int > best_val_acc_int:
      best_val_acc_int = val_acc_int
      test_acc_int = tmp_test_acc_int
      best_epoch_int = epoch
    # store best stuff at the end of integrator forward pass
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, tmp_test_acc))
    log = 'Performance inside integrator Val: {:.4f}, Test: {:.4f}'
    print(log.format(val_acc_int, tmp_test_acc_int))
    # print(
    # log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))
  print('best in integrator val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc_int,
                                                                                                test_acc_int,
                                                                                                best_epoch_int))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                      help='use the 10 fixed splits from '
                           'https://arxiv.org/abs/2002.05287')
  parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                      help='the number of splits to repeat the results on')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument('--planetoid_split', action='store_true',
                      help='use planetoid splits for Cora/Citeseer/Pubmed')
  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, default=0.3, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, default='hyperbolic_laplacian', help='hyperbolic_laplacian, laplacian, transformer, dorsey, GAT')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

  # ODE args
  parser.add_argument('--time', type=float, default=2.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, default = 'hyperbolic_rk4', help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=0.5,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=200, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=0.5, help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=128,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
  # beltrami args
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
  parser.add_argument('--pos_enc_type', type=str, default="DW64",
                      help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                      help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention",
                      help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  parser.add_argument('--symmetric_attention', action='store_true',
                      help='maks the attention symmetric for rewring in QK space')

  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")


  args = parser.parse_args()

  opt = vars(args)

  main(opt)