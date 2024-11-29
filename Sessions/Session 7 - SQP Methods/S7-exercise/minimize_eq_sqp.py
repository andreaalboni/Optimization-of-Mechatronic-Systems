import numpy as np
from finite_difference_jacobian import finite_difference_jacobian
from merit_functions import merit_function_equality_constrained_sqp as merit_function
from line_search import line_search
import casadi as cs


def minimize_eq_sqp(ffun, hfun, x0: np.array, with_line_search=True, with_powells_trick=False):

    # convergence tolerance
    grad_tol = 1e-4
    max_iters = 100
    
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
    
    # a log of the iterations
    x_iters = np.zeros((n_variables, max_iters))
    gradient_lagrangian_iters = np.zeros((1,max_iters))
    
    # initial variables
    x = x0.copy()
    lambdA = np.zeros((n_eq_constraints, 1)) # lambda is a key word
    
    # initialize B
    B = np.eye(n_variables)
    
    # evaluate the initial f and it's jacobian J
    _, J = finite_difference_jacobian(ffun, x)
    
    # evaluate the initial h and it's jacobian J_h
    h, J_h = finite_difference_jacobian(hfun, x)

    # Define the QP solver with Casadi
    B_placeholder = cs.DM.ones(n_variables, n_variables)
    Jh_placeholder = cs.DM.ones(n_eq_constraints, n_variables)
    qp_struct = {   'h': B_placeholder.sparsity(),
                    'a': Jh_placeholder.sparsity()}
            
    qp_solver_opts = {}
    qp_solver_opts["print_out"] = False
    qp_solver_opts["osqp"] = {"verbose": False,
                              "eps_abs":1e-4,
                              "eps_rel":1e-4,
                              "eps_prim_inf":1e-5,
                              "eps_dual_inf":1e-5,
                              "max_iter":4000}
    
    qp_solver = cs.conic("qpsol", "osqp", qp_struct, qp_solver_opts)

    # Main optimization loop
    for k in range(max_iters):
        
        # check for divergence
        x_norm_inf = np.linalg.norm(x, np.inf)
        if x_norm_inf > 1e6:
            raise ValueError('minimize_sqp has diverged, ||x||_\{inf\}: %.3g',x_norm_inf)
        
        # store x in the iteration log
        gradient_lagrangian = J.T + J_h.T  @ lambdA
        norm_gradient_lagrangian = np.linalg.norm(gradient_lagrangian, np.inf)
        norm_infeasbility = np.linalg.norm(h, np.inf)

        norm_convergence = np.fmax(norm_gradient_lagrangian, norm_infeasbility)

        x_iters[:,[k]] = x.copy()
        gradient_lagrangian_iters[0, k] = norm_convergence
        
        print('iteration: {},   convergence_metric: {:.2e}'.format(k, norm_convergence))
    
        # check for convergence
        if norm_convergence < grad_tol:
            x_iters = x_iters[:,:k+1]
            gradient_lagrangian_iters = gradient_lagrangian_iters[:,:k+1]
            print('acceptable solution found\n')
            return x, x_iters, gradient_lagrangian_iters
        
        # lambdA_old = lambdA.copy()
        x_old = x.copy()
        
        #################### FILL THIS PART IN ########################
        # find the search direction and lambda

        ### OLD SCHOOL WITH SOLVING LINEAR SYSTEM -----------------------------
        # M = np.vstack((np.hstack((B, J_h.T)), 
        #                np.hstack((J_h,  np.zeros((n_eq_constraints, n_eq_constraints))))
        #                ))
        # L0 = np.vstack((-J.T, -h))
        # sol = np.linalg.solve(M, L0)
        # dk = sol[:n_variables]
        # lambdA = sol[n_variables:]

        ### NEW APPROACH SOLVING QP
        raise NotImplementedError("TODO: Implement the QP solver")
        # lba = ?????
        # uba = ?????
        # result = qp_solver(h=????,
        #                    a=???,
        #                    g=???,
        #                    lba=???,
        #                    uba=???)
        # dk = np.array(????)
        # lambdA = np.array(????)
        
        ### END OF DIFFERENT APPROACHES ---------------------------------------
        # take the line search
        if with_line_search:
            sigma = 0.01
            beta  = 0.6
            c     = 100
            # calculate the directional derivative
            directional_derivative = J @ dk - c*np.linalg.norm(h,1)
            merit_fun = lambda y : merit_function(ffun, hfun, c, y)
            x, _ = line_search(merit_fun, x, directional_derivative, dk, sigma, beta)
        else:
            x += dk

        # ------ BFGS Hessian Approximation ------
        J_old = J.copy()
        J_h_old = J_h.copy()
        
        # evaluate F and it's jacobian J
        _, J = finite_difference_jacobian(ffun, x)
    
        # evaluate h and it's jacobian J_h
        h, J_h = finite_difference_jacobian(hfun, x)
        
        Lx_old = J_old.T + J_h_old.T @ lambdA
        Lx     = J.T     + J_h.T @ lambdA
        
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

        B = B - B @ (s@s.T) @ B/(s.T@B@s) + y@y.T/(s.T @ y)

        # Symmetrize Hessian (only needed for numerical accuracy)
        B = (B + B.T) / 2. # Symmetrize Hessian, only needed due to numerical inaccuracy, so that quadprog doesn't complain
        
        ###############################################################    

    raise RuntimeError('minimize_eq_sqp: max iterations exceeded')

