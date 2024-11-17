import numpy as np
from finite_difference_jacobian import finite_difference_jacobian
from merit_functions import merit_function_inequality_constrained_sqp as merit_function
from line_search import line_search
import casadi as cs

def minimize_sqp(ffun, hfun, gfun, x0: np.array, with_line_search=True, with_powells_trick=False, callback=None):

    # convergence tolerance
    convergence_tol = 1e-4
    max_iters = 1000
    
    # make sure x0 is a column vector
    n_variables, cols = x0.shape
    if cols != 1:
        raise ValueError('x0 needs to be a column vector')
    
    # make sure ffun returns a scalar
    f0 = ffun(x0)
    if len(f0) != 1:
        raise ValueError('ffun must return a scalar')
    
    # make sure hfun returns a column vector
    h0 = hfun(x0)
    n_eq_constraints, cols = h0.shape
    if cols != 1:
        raise ValueError('hfun needs to return a column vector')
    
    # make sure gfun returns a column vector
    g0 = gfun(x0)
    n_ineq_constraints, cols = g0.shape
    if cols != 1:
        raise ValueError('gfun needs to return a column vector')
    
    n_constraints = n_eq_constraints + n_ineq_constraints
    # a log of the iterations
    x_iters = np.zeros((n_variables, max_iters))
    gradient_lagrangian_iters = np.zeros((1,max_iters))
    
    # initial variables
    x      = x0.copy()
    lambdA = np.zeros((n_eq_constraints, 1)) # lambda is a key word
    mu     = np.zeros((n_ineq_constraints, 1))
    
    # set these so the first printout doesn't fail
    dk     = np.zeros((n_variables, 1))
    alpha = 0
    qp_status = "default"
    
    # initialize B
    B = np.eye(n_variables)
    
    # evaluate the initial f and it's jacobian J
    _, J = finite_difference_jacobian(ffun, x)
    
    # evaluate the initial h and it's jacobian J_h
    h, J_h = finite_difference_jacobian(hfun, x)
    
    # evaluate the initial g and it's jacobian J_g
    g, J_g = finite_difference_jacobian(gfun, x)
    
    # Define the QP solver with Casadi
    B_placeholder = cs.DM.ones(n_variables, n_variables)
    Jh_placeholder = cs.DM.ones(n_constraints, n_variables)
    qp_struct = {   'h': B_placeholder.sparsity(),
                    'a': Jh_placeholder.sparsity()}
    

    qp_solver_opts = {}
    qp_solver_opts["print_out"] = False
    qp_solver_opts["printLevel"] = "none"

    qp_solver = cs.conic("qpsol", "qpoases", qp_struct, qp_solver_opts)

    # Main optimization loop
    for k in range(max_iters):
        
        # check for divergence
        x_norm_inf = np.linalg.norm(x, np.inf)
        if x_norm_inf > 1e6:
            raise ValueError('minimize_sqp has diverged, ||x||_\{inf\}: %.3g',x_norm_inf)
        
        # check for bad BFGS update
        if np.any(np.any(np.isnan(B))):
            raise ValueError('BFGS has NaNs in it, step size is probably very small')
        
        # store x in the iteration log
        gradient_lagrangian = J.T + J_h.T  @ lambdA + J_g.T @ mu
        norm_gradient_lagrangian = np.linalg.norm(gradient_lagrangian, np.inf)
        norm_infeasbility = np.linalg.norm( np.vstack((h, np.fmax(g,0))), np.inf)
    
        norm_convergence = np.fmax(norm_gradient_lagrangian, norm_infeasbility)

        x_iters[:,[k]] = x.copy()
        gradient_lagrangian_iters[0, k] = norm_convergence
        
        # Iteration output
        if k % 10 == 0:
            print('{0: >8}{1: >20}{2: >20}{3: >30}{4: >15}{5: >7}'.format("iter", "qpstatus", "||grad_L||", "||infeasibility_measure||", "||step||", "t"))
        print('{0: >8d}{1: >20}{2: >20.4e}{3: >30.4e}{4: >15.4e}{5: >7.4f}'.format(k, qp_status, norm_gradient_lagrangian, norm_infeasbility, np.linalg.norm(dk,np.inf), alpha))
        
        if callback is not None:
            callback(x)

        # check for convergence
        if norm_convergence < convergence_tol:
            x_iters = x_iters[:,:k+1]
            gradient_lagrangian_iters = gradient_lagrangian_iters[:,:k+1]
            print('acceptable solution found\n')
            return x, x_iters, gradient_lagrangian_iters
        
        x_old = x.copy()
        
        #################### FILL THIS PART IN ########################
        # find the search direction and lambda
        raise NotImplementedError("TODO: Implement the QP solver")
        # lba = ????
        # uba = ????
        
        # result = qp_solver(h=????,
        #                    a=????,
        #                    g=????,
        #                    lba=????,
        #                    uba=????)

        # dk = np.array(????)
        # lambdA = np.array(???)[?????]
        # mu = np.array(???)[?????]
        # qp_status = qp_solver.stats()['return_status']
        
    
        ### END OF DIFFERENT APPROACHES ---------------------------------------
        # take the line search
        if with_line_search:
            sigma = 0.01
            beta  = 0.6
            c     = 100

            raise NotImplementedError("TODO: Implement the directional derivative")
            # directional_derivative = ?????
            merit_fun = lambda y : merit_function(ffun, hfun, gfun, c, y)
            x, alpha = line_search(merit_fun, x, directional_derivative, dk, sigma, beta)
        else:
            x += dk

        # update BFGS hessian approximation
        J_old   = J.copy()
        J_h_old = J_h.copy()
        J_g_old = J_g.copy()
        
        # evaluate F and it's jacobian J
        _, J = finite_difference_jacobian(ffun, x)
    
        # evaluate h and it's jacobian J_h
        h, J_h = finite_difference_jacobian(hfun, x)
        
        # evaluate g and it's jacobian J_g
        g, J_g = finite_difference_jacobian(gfun, x)
        
        raise NotImplementedError("TODO: Implement the Lagrangian gradients!")
        # Lx_old = ???
        # Lx     = ???
        
        s = x - x_old
        y = Lx - Lx_old

        # Powell's trick
        if with_powells_trick:
            if y.T @ s >= 0.2 * s.T @ B @ s:
                theta = 1
            else:
                theta = (0.8 * s.T @ B @ s)/(s.T @ B @ s -s.T @ y)

            y = theta*y + (1-theta) * B @ s
        else:
            y = y
        
        B = B - B@(s@s.T)@B/(s.T @B@s) + y@y.T/(s.T@y)
        B = (B + B.T) / 2 # Symmetrize Hessian, only needed due to numerical inaccuracy, so that quadprog doesn't complain
        ###############################################################
    
    raise RuntimeError('minimize_sqp: max iterations exceeded')

