import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


def library_size(n_x, n_u, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n_x+k-1,k))

    if use_sine:
        l += 2*n_x
        l += 2*n_x*n_u
    if not include_constant:
        l -= 1

    l += n_u
    if poly_order > 1:
        l += binom(n_u+1,2)
    return l


def sindy_library(X, U, poly_order, include_sine=False):
    m_x,n_x = X.shape
    m_u,n_u = U.shape
    l = library_size(n_x, n_u, poly_order, include_sine, True)
    library = np.ones((m_x,l))
    index = 1

    for i in range(n_x):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n_x):
            for j in range(i,n_x):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n_x):
            for j in range(i,n_x):
                for k in range(j,n_x):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n_x):
            for j in range(i,n_x):
                for k in range(j,n_x):
                    for q in range(k,n_x):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1

    if poly_order > 4:
        for i in range(n_x):
            for j in range(i,n_x):
                for k in range(j,n_x):
                    for q in range(k,n_x):
                        for r in range(q,n_x):
                                library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                                index += 1

    if include_sine:
        for i in range(n_x):
            library[:,index] = np.sin(X[:,i])
            index += 1
            library[:,index] = np.cos(X[:,i])
            index += 1

    for i in range(n_u):
        library[:,index] = U[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n_u):
            for j in range(i,n_u):
                library[:,index] = U[:,i]*U[:,j]
                index += 1
        for i in range(n_x):
            for j in range(n_u):
                library[:,index] = X[:,i]*U[:,j]
                index += 1
    if include_sine:
        for i in range(n_x):
            for j in range(n_u):
                library[:,index] = U[:,j]*np.sin(X[:,i])
                index += 1
                library[:,index] = U[:,j]*np.cos(X[:,i])
                index += 1

    return library


def sindy_library_order2(X, dX, U, poly_order, include_sine=False):
    X_combined = np.concatenate((X, dX), axis=1)
    return sindy_library(X_combined, U, poly_order, include_sine=False)


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]

    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, U, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), U(t), poly_order, include_sine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, U, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]

    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x
