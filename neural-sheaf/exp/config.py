OPTIMIZATION_DEFAULTS = {
    'epochs': 150,
    'lr': 0.005,
    'weight_decay': 1e-2,
    'sheaf_decay': 1e-2,           # Weight decay for sheaf-map parameters.
    'early_stopping': 40,
    'grad_clip': 2.0,              # Gradient-norm clipping threshold.
    'batch_size': 32,
}

EVALUATION_DEFAULTS = {
    'seed': 42,
    'num_folds': 5,
    'num_repeats': 10,             # Independent five-fold repetitions.
    'validation_fraction': 0.2,
}

RUNTIME_DEFAULTS = {
    'amp': 'auto',                 # CUDA mixed precision when available.
    'num_workers': 0,              # DataLoader workers
}

MODEL_DEFAULTS = {
    'layers': 2,                   # Number of hyperbolic ODE blocks.
    'd': 2,                        # Sheaf stalk dimension.
    'hidden_channels': 256,        # Hidden representation width.
    'input_dropout': 0.5,          # Dropout before the first ODE block.
    'dropout': 0.5,                # Dropout between ODE blocks.
    'left_weights': True,          # Enable left linear ODE transform.
    'right_weights': True,         # Enable right linear ODE transform.
    'use_act': True,               # Apply the ODE activation.
    'second_linear': False,        # Do not add a second input projection.
    'sheaf_act': 'tanh',           # Activation in the sheaf-map learner.
    'pool_type': 'mean',           # Graph-level readout.
    'use_edge_features': True,     # Feed edge attributes to the sheaf learner.
    'use_learned_sheaf': True,     # Learn restriction maps from graph features.
    'use_fixed_hodge': False,      # a fixed Hodge correction.
}

ODE_DEFAULTS = {
    'max_t': 0.1,                  # End time of ODE integration.
    'int_method': 'hyperbolic_rk4',# Fixed-step hyperbolic ODE solver.
    'step_size': 1.0,              # Solver step size.
    'curvature': 1.0,              # Poincare-ball curvature.
    'normalised': True,            # Use normalized sheaf Laplacian.
    'deg_normalised': False,       # Do not apply extra degree normalization.
    'max_nfe': 5000,               # ODE function-evaluation budget.
}

FORMAL_DEFAULTS = {
    **OPTIMIZATION_DEFAULTS,
    **EVALUATION_DEFAULTS,
    **RUNTIME_DEFAULTS,
    **MODEL_DEFAULTS,
    **ODE_DEFAULTS,
}
