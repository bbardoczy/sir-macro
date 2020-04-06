"""
Solving the SIR-Macro model of Eichenbaum-Rebelo-Trabandt (2020): The Macroeconomics of Epidemics

Author: Bence Bardoczy, Northwestern University
Date: 3/27/2020
"""

# import standard python packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# import supporting routines that are not specific to this model
import utilities as utils


'''Part 1: Transition dynamics of epidemic'''


def initial_ss(A=39.8, beta=0.96**(1/52), theta=36):
    """Pre-epidemic steady state."""
    w = A
    N = theta ** (-0.5)
    C = A * N
    U = (np.log(C) - theta/2 * N ** 2) / (1 - beta)
    return {'w': w, 'C': C, 'N': N, 'ns': N, 'ni': N, 'nr': N, 'U': U, 'A': A, 'beta': beta, 'theta': theta, 'Neff': N}


def td_sir(pi3=0.6165, H=250, eps=0.001, pid=0.07/18, pir=0.99*7/18, phi=0.8, theta=36, A=39.8):
    """Mechanical SIR model. Assumes no containment policy. Produces same result td_solve with pi1=pi2=0."""
    # initialize population shares
    S, I, R, D = np.empty(H), np.empty(H), np.empty(H), np.empty(H)
    S[0], I[0], R[0], D[0] = 1 - eps, eps, 0, 0
    T = np.zeros(H)

    # trace out pandemic
    for t in range(1, H):
        # transmissions last week
        T[t-1] = pi3 * S[t-1] * I[t-1]

        # this week
        S[t] = S[t-1] - T[t-1]
        I[t] = I[t-1] + T[t-1] - (pir + pid) * I[t-1]
        R[t] = R[t-1] + pir * I[t-1]
        D[t] = D[t-1] + pid * I[t-1]

    # population
    P = np.ones(H)
    P -= D

    # everybody chooses the steady state hours, consumption depends on productivity (relies on no containment)
    n = theta ** (-0.5)
    cs, cr = A * n, A * n
    ci = phi * A * n

    # aggregates
    N = n * (S + I + R)
    C = S * cs + I * ci + R * cr

    return {'S': S, 'I': I, 'R': R, 'D': D, 'P': P, 'N': N, 'C': C}


def td_eval(ns, ni, nr, ctax, U_ss, c_ss, n_ss, pr_treat=np.zeros(250), pr_vacc=np.zeros(250),
            pi1=0.0046, pi2=7.3983, pi3=0.2055, H=250, eps=0.001, pidbar=0.07/18, pir=0.99*7/18, phi=0.8, theta=36,
            A=39.8, beta=0.96**(1/52), kappa=0.0):
    """Evaluates SIR-Macro model for a given guess of employment sequences, conditional on a containment policy."""

    # 1. consumption
    cr = A / ((1 + ctax) * theta * nr)
    transfer = (1 + ctax) * cr - A * nr
    ci = (A * phi * ni + transfer) / (1 + ctax)
    cs = (A * ns + transfer) / (1 + ctax)

    # 2. population shares
    S, I, R, D, pid = np.empty(H), np.empty(H), np.empty(H), np.empty(H), np.empty(H)
    S[0], I[0], R[0], D[0] = 1 - eps, eps, 0, 0
    T = np.zeros(H)
    for t in range(1, H):
        # transmissions last week
        T[t-1] = pi1 * S[t-1] * cs[t-1] * I[t-1] * ci[t-1] + pi2 * S[t-1] * ns[t-1] * I[t-1] * ni[t-1] +\
                 pi3 * S[t-1] * I[t-1]

        # this week
        pid[t-1] = pidbar + kappa * I[t-1] ** 2
        S[t] = S[t-1] - T[t-1]
        I[t] = I[t-1] + T[t-1] - (pir + pid[t-1]) * I[t-1]
        R[t] = R[t-1] + pir * I[t-1]
        D[t] = D[t-1] + pid[t-1] * I[t-1]

    # population
    P = np.ones(H)
    P -= D

    # for completeness
    pid[-1] = pidbar + kappa * I[-1] ** 2

    # 3. value functions (we know that policy functions are the same in initial and terminal ss)
    tau = pi1 * cs * I * ci + pi2 * ns * I * ni + pi3 * I
    Ur, Ui, Us = np.empty(H+1), np.empty(H+1), np.empty(H+1)
    Ur[-1], Us[-1] = U_ss, U_ss
    Ui[-1] = (np.log(phi * c_ss) - theta / 2 * n_ss ** 2 + (1 - pr_treat[-1]) * beta * pir * Ur[-1] +
              beta * pr_treat[-1] * Ur[-1]) / (1 - beta * (1 - pir - pid[-1]))
    for t in reversed(range(H)):
        Ur[t] = np.log(cr[t]) - theta / 2 * nr[t] ** 2 + beta * Ur[t+1]
        Ui[t] = np.log(ci[t]) - theta / 2 * ni[t] ** 2 + (1 - pr_treat[t]) * beta * ((1 - pir - pid[t]) * Ui[t+1] +
                    pir * Ur[t+1]) + beta * pr_treat[t] * Ur[t+1]
        Us[t] = np.log(cs[t]) - theta / 2 * ns[t] ** 2 + (1 - pr_vacc[t]) * beta * ((1 - tau[t]) * Us[t+1] +
                    tau[t] * Ui[t+1]) + pr_vacc[t] * beta * Ur[t+1]

    # 4. multipliers
    mus = beta * (1 - pr_vacc) * (Us[1:] - Ui[1:])
    lams = (theta * ns + mus * pi2 * I * ni) / A
    lami = theta * ni / (phi * A)

    # 5. residuals
    R1 = ctax * (S * cs + I * ci + R * cr) - transfer * (S + I + R)
    R2 = lami * (1 + ctax) - 1 / ci
    R3 = lams * (1 + ctax) + mus * pi1 * I * ci - 1 / cs

    # 6. extras
    C = S * cs + I * ci + R * cr
    N = S * ns + I * ni + R * nr
    Neff = S * ns + I * phi * ni + R * nr  # effective hours
    walras = A * Neff - C
    mortality = pid / pir

    return {'ns': ns, 'ni': ni, 'nr': nr, 'cs': cs, 'ci': ci, 'cr': cr, 'transfer': transfer, 'T': T, 'S': S, 'I': I,
            'R': R, 'D': D, 'tau': tau, 'Ur': Ur[:-1], 'Ui': Ui[:-1], 'Us': Us[:-1], 'mus': mus, 'lami': lami, 'P': P,
            'lams': lams, 'R1': R1, 'R2': R2, 'R3': R3, 'C': C, 'N': N, 'Neff': Neff, 'walras': walras, 'pid': pid,
            'pr_treat': pr_treat, 'pr_vacc': pr_vacc, 'mortality': mortality, 'ctax': ctax,
            'kappa': kappa, 'pir': pir, 'beta': beta, 'pidbar': pidbar, 'theta': theta, 'A': A, 'eps': eps, 'pi1': pi1,
            'pi2': pi2, 'pi3': pi3}


def get_J(ss, ctax, pr_treat=np.zeros(250), pr_vacc=np.zeros(250), pi1=0.0046, pi2=7.3983, pi3=0.2055, H=250,
          eps=0.001, pidbar=0.07/18, pir=0.99*7/18, phi=0.8, theta=36, A=39.8, beta=0.96**(1/52), kappa=0.0, h=1E-4):
    """Compute Jacobian at initial/terminal steady state of unknowns. Uses forward difference."""
    # compute Jacobian around initial ss (unknowns will return to same ss)
    td0 = td_eval(ns=ss['N'] * np.ones(H), ni=ss['N'] * np.ones(H), nr=ss['N'] * np.ones(H),
                  pr_treat=pr_treat, pr_vacc=pr_vacc, U_ss=ss['U'],
                  ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir, kappa=kappa,
                  phi=phi, theta=theta, A=A, beta=beta, c_ss=ss['C'], n_ss=ss['N'])

    # initialize Jacobian
    J = dict()
    for o in td0.keys():
        J[o] = dict()
        for i in ['ns', 'ni', 'nr']:
            J[o][i] = np.empty((H, H))

    # Compute via direct method
    for t in range(H):
        # unknown 1: ns
        td_ns = td_eval(ns=ss['N'] + h * (np.arange(H) == t), ni=ss['N'] * np.ones(H), nr=ss['N'] * np.ones(H),
                        U_ss=ss['U'], ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, kappa=kappa,
                        pidbar=pidbar, pir=pir, phi=phi, theta=theta, A=A, beta=beta, c_ss=ss['C'], n_ss=ss['N'],
                        pr_treat=pr_treat, pr_vacc=pr_vacc)

        # unknown 2: ni
        td_ni = td_eval(ns=ss['N'] * np.ones(H), ni=ss['N'] + h * (np.arange(H) == t), nr=ss['N'] * np.ones(H),
                        U_ss=ss['U'], ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir,
                        phi=phi, theta=theta, A=A, beta=beta, c_ss=ss['C'], n_ss=ss['N'], kappa=kappa,
                        pr_treat=pr_treat, pr_vacc=pr_vacc)

        # unknown 3: nr
        td_nr = td_eval(ns=ss['N'] * np.ones(H), ni=ss['N'] * np.ones(H), nr=ss['N'] + h * (np.arange(H) == t),
                        U_ss=ss['U'], ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir,
                        phi=phi, theta=theta, A=A, beta=beta, c_ss=ss['C'], n_ss=ss['N'], kappa=kappa,
                        pr_treat=pr_treat, pr_vacc=pr_vacc)

        # jacobian as nested dict
        for o in td_ns.keys():
            J[o]['ns'][:, t] = (td_ns[o] - td0[o]) / h
            J[o]['ni'][:, t] = (td_ni[o] - td0[o]) / h
            J[o]['nr'][:, t] = (td_nr[o] - td0[o]) / h

    return J


def td_solve(ctax, pr_treat=np.zeros(250), pr_vacc=np.zeros(250), pi1=0.0046, pi2=7.3983, pi3=0.2055, eps=0.001,
             pidbar=0.07 / 18, pir=0.99 * 7 / 18, kappa=0.0, phi=0.8, theta=36, A=39.8, beta=0.96**(1/52), maxit=50,
             h=1E-4, tol=1E-8, noisy=False, H_U=None):
    """Solve SIR-macro model via Newton's method."""
    # infer length from guess
    H = ctax.shape[0]
    unknowns = ['ns', 'ni', 'nr']
    targets = ['R1', 'R2', 'R3']

    # compute initial ss
    ss = initial_ss(A=A, beta=beta, theta=theta)

    # compute jacobian
    if H_U is None:
        print('Precomputing Jacobian...')
        J = get_J(ss=ss, ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir, phi=phi,
                  theta=theta, A=A, beta=beta, h=h, kappa=kappa, pr_treat=pr_treat, pr_vacc=pr_vacc)
        H_U = utils.J_to_HU(J, H, unknowns, targets)
        print('Done!')

    # initialize guess for unknowns to steady state length T
    Us = {k: np.full(H, ss[k]) for k in unknowns}
    Uvec = utils.pack_vectors(Us, unknowns, H)

    # iterate until convergence
    for it in range(maxit):
        # evaluate function
        results = td_eval(**Us, U_ss=ss['U'], ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar,
                          pir=pir, phi=phi, theta=theta, A=A, beta=beta, c_ss=ss['C'], n_ss=ss['N'], kappa=kappa,
                          pr_treat=pr_treat, pr_vacc=pr_vacc)
        errors = {k: np.max(np.abs(results[k])) for k in targets}

        # update guess
        if noisy:
            print(f'On iteration {it}')
            for k in errors:
                print(f'   max error for {k} is {errors[k]:.2E}')
        if all(v < tol for v in errors.values()):
            if noisy:
                print(f'Convergence after {it} iterations!')
            break
        else:
            Hvec = utils.pack_vectors(results, targets, H)
            Uvec -= utils.factored_solve(H_U, Hvec)
            Us = utils.unpack_vectors(Uvec, unknowns, H)
    else:
        raise ValueError(f'No convergence after {maxit} iterations!')

    return results


'''
Part 2: Optimal policy

Computing the Ramsey policy takes a few minutes. I have found that 1E-3 is sufficient precision.
'''


def planner(ctax, s0=1, i0=1, r0=1, pr_treat=np.zeros(250), pr_vacc=np.zeros(250), pi1=0.0046, pi2=7.3983, pi3=0.2055,
            eps=0.001, pidbar=0.07 / 18, pir=0.99 * 7 / 18, kappa=0.0, phi=0.8, theta=36, A=39.8, beta=0.96**(1/52),
            maxit=100, h=1E-4, tol=1E-8, noisy=False, H_U=None):
    """Objective function."""

    # solve transition path for given guess
    out = td_solve(ctax, pr_treat=pr_treat, pr_vacc=pr_vacc, pi1=pi1, pi2=pi2, pi3=pi3, eps=eps, pidbar=pidbar, pir=pir,
                   kappa=kappa, phi=phi, theta=theta, A=A, beta=beta, maxit=maxit, h=h, tol=tol, noisy=noisy, H_U=H_U)

    # welfare
    W = s0 * out['S'][0] * out['Us'][0] + i0 * out['I'][0] * out['Ui'][0] + r0 * out['R'][0] * out['Ur'][0]

    return -W


def planner_jac(ctax, s0=1, i0=1, r0=1, pr_treat=np.zeros(250), pr_vacc=np.zeros(250), pi1=0.0046, pi2=7.3983,
                pi3=0.2055, eps=0.001, pidbar=0.07 / 18, pir=0.99 * 7 / 18, kappa=0.0, phi=0.8, theta=36, A=39.8,
                beta=0.96**(1/52), maxit=50, h=1E-4, tol=1E-8, noisy=False, H_U=None):
    """Differentiate planner function."""
    # 1. precompute Jacobian for solving equilibrium
    if H_U is None:
        print('Precomputing Jacobian...')
        ss = initial_ss(A=A, beta=beta, theta=theta)
        H = ctax.shape[0]
        J = get_J(ss=ss, ctax=ctax, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir, phi=phi,
                  theta=theta, A=A, beta=beta, h=h, kappa=kappa, pr_treat=pr_treat, pr_vacc=pr_vacc)
        H_U = utils.J_to_HU(J, H=H, unknowns=['ns', 'ni', 'nr'], targets=['R1', 'R2', 'R3'])
        print('Done!')

    # 2. welfare at current policy
    W0 = planner(ctax=ctax, s0=s0, i0=i0, r0=r0, pr_treat=pr_treat, pr_vacc=pr_vacc, pi1=pi1, pi2=pi2, pi3=pi3,
                 eps=eps, pidbar=pidbar, pir=pir, kappa=kappa, phi=phi, theta=theta, A=A, beta=beta, maxit=maxit, h=h,
                 tol=tol, noisy=noisy, H_U=H_U)

    # 3. perturb policy period by period
    H = ctax.shape[0]
    dW = np.zeros(H)
    for t in range(H):
        W1 = planner(ctax=ctax + h * (np.arange(H) == t), s0=s0, i0=i0, r0=r0, pr_treat=pr_treat,
                     pr_vacc=pr_vacc,
                     pi1=pi1, pi2=pi2, pi3=pi3, eps=eps, pidbar=pidbar, pir=pir, kappa=kappa, phi=phi, theta=theta,
                     A=A, beta=beta, maxit=maxit, h=h, tol=tol, noisy=noisy, H_U=H_U)
        dW[t] = (W1 - W0) / h

    return dW


def ramsey(ctax0=np.zeros(250), s0=1, i0=1, r0=1, pr_treat=np.zeros(250), pr_vacc=np.zeros(250), pi1=0.0046,
           pi2=7.3983, pi3=0.2055, eps=0.001, pidbar=0.07 / 18, pir=0.99 * 7 / 18, kappa=0.0, phi=0.8, theta=36, A=39.8,
           beta=0.96**(1/52), maxit=100, h=1E-4, tol=1E-8, tol_ramsey=1E-3, noisy=False):
    """Brute-force maximization"""
    # 1. precompute Jacobian for solving equilibrium and computing Jacobian EVERY TIME
    print('Precomputing Jacobian...')
    ss = initial_ss(A=A, beta=beta, theta=theta)
    H = ctax0.shape[0]
    J = get_J(ss=ss, ctax=ctax0, pi1=pi1, pi2=pi2, pi3=pi3, H=H, eps=eps, pidbar=pidbar, pir=pir, phi=phi,
              theta=theta, A=A, beta=beta, h=h, kappa=kappa, pr_treat=pr_treat, pr_vacc=pr_vacc)
    H_U = utils.J_to_HU(J, H=H, unknowns=['ns', 'ni', 'nr'], targets=['R1', 'R2', 'R3'])
    print('Done!')

    # objective
    obj = lambda ctax: planner(ctax, s0=s0, i0=i0, r0=r0, pr_treat=pr_treat, pr_vacc=pr_vacc, pi1=pi1, pi2=pi2, pi3=pi3,
                               eps=eps, pidbar=pidbar, pir=pir, kappa=kappa, phi=phi, theta=theta, A=A, beta=beta,
                               maxit=maxit, h=h, tol=tol, noisy=noisy, H_U=H_U)

    # derivative
    jac = lambda ctax: planner_jac(ctax, s0=s0, i0=i0, r0=r0, pr_treat=pr_treat, pr_vacc=pr_vacc, pi1=pi1,
                                   pi2=pi2, pi3=pi3, eps=eps, pidbar=pidbar, pir=pir, kappa=kappa, phi=phi, theta=theta,
                                   A=A, beta=beta, maxit=maxit, h=h, tol=tol, noisy=noisy, H_U=H_U)

    # run
    res = opt.minimize(obj, jac=jac, x0=ctax0, method='BFGS', tol=tol_ramsey, options={'disp': True})

    if res.success:
        print('Success!')
    else:
        print('Fail!')

    return res


'''Part 3: Replication'''


def fig12(h=150, H=250):
    """SIR-Macro Model vs. SIR Model"""
    ss = initial_ss()
    td1 = td_sir()
    td2 = td_solve(ctax=np.full(H, 0))

    fig1, axes = plt.subplots(2, 3, figsize=(.8*12, .8*8))
    ax = axes.flatten()

    ax[0].plot(100 * td1['I'][:h], label='SIR', linewidth=2)
    ax[0].plot(100 * td2['I'][:h], label='SIR-macro', linewidth=2)
    ax[0].set_title('Infected, I')
    ax[0].set_ylabel('% of initial population')
    ax[0].legend()

    ax[1].plot(100 * td1['S'][:h], label='SIR', linewidth=2)
    ax[1].plot(100 * td2['S'][:h], label='SIR-macro', linewidth=2)
    ax[1].set_title('Susceptibles, S')
    ax[1].set_ylabel('% of initial population')
    ax[1].legend()

    ax[2].plot(100 * td1['R'][:h], label='SIR', linewidth=2)
    ax[2].plot(100 * td2['R'][:h], label='SIR-macro', linewidth=2)
    ax[2].set_title('Recovered, R')
    ax[2].set_ylabel('% of initial population')
    ax[2].legend()

    ax[3].plot(100 * td1['D'][:h], label='SIR', linewidth=2)
    ax[3].plot(100 * td2['D'][:h], label='SIR-macro', linewidth=2)
    ax[3].set_title('Deaths, D')
    ax[3].set_ylabel('% of initial population')
    ax[3].set_xlabel('weeks')
    ax[3].legend()

    ax[4].plot(100 * (td1['C'] / ss['C'] - 1)[:h], label='SIR', linewidth=2)
    ax[4].plot(100 * (td2['C'] / ss['C'] - 1)[:h], label='SIR-macro', linewidth=2)
    ax[4].axhline(0, color='gray', linestyle='--')
    ax[4].set_title('Aggregate Consumption, C')
    ax[4].set_ylabel('% deviation from initial ss')
    ax[4].set_xlabel('weeks')
    ax[4].legend()

    ax[5].plot(100 * (td1['N'] / ss['N'] - 1)[:h], label='SIR', linewidth=2)
    ax[5].plot(100 * (td2['N'] / ss['N'] - 1)[:h], label='SIR-macro', linewidth=2)
    ax[5].axhline(0, color='gray', linestyle='--')
    ax[5].set_title('Aggregate Hours, N')
    ax[5].set_ylabel('% deviation from initial ss')
    ax[5].set_xlabel('weeks')
    ax[5].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    # individual policies
    fig2, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.flatten()

    ax[0].plot(100 * (td2['cs'] / ss['C'] - 1)[:h], label='susceptible', linewidth=2, linestyle='-')
    ax[0].plot(100 * (td2['ci'] / ss['C'] - 1)[:h], label='infected', linewidth=2, linestyle='--')
    ax[0].plot(100 * (td2['cr'] / ss['C'] - 1)[:h], label='recovered', linewidth=2, linestyle='-.')
    ax[0].set_title('Consumption by Type')
    ax[0].set_ylabel('% deviation from initial ss')
    ax[0].set_xlabel('weeks')
    ax[0].legend()

    ax[1].plot(100 * (td2['ns'] / ss['N'] - 1)[:h], label='susceptible', linewidth=2, linestyle='-')
    ax[1].plot(100 * (td2['ni'] / ss['N'] - 1)[:h], label='infected', linewidth=2, linestyle='--')
    ax[1].plot(100 * (td2['nr'] / ss['N'] - 1)[:h], label='recovered', linewidth=2, linestyle='-.')
    ax[1].set_title('Hours by Type')
    ax[1].set_ylabel('% deviation from initial ss')
    ax[1].set_xlabel('weeks')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def fig4(h=150, H=250):
    """Medical Preparedness"""
    ss = initial_ss()
    td1 = td_solve(ctax=np.full(H, 0))
    td2 = td_solve(ctax=np.full(H, 0), kappa=12.5)

    fig1, axes = plt.subplots(3, 3, figsize=(.9 * 12, .9 * 12))
    ax = axes.flatten()

    ax[0].plot(100 * td1['I'][:h], label='baseline', linewidth=2)
    ax[0].plot(100 * td2['I'][:h], label='medical capacity', linewidth=2)
    ax[0].set_title('Infected, I')
    ax[0].set_ylabel('% of initial population')
    ax[0].legend()

    ax[1].plot(100 * td1['S'][:h], label='baseline', linewidth=2)
    ax[1].plot(100 * td2['S'][:h], label='medical capacity', linewidth=2)
    ax[1].set_title('Susceptibles, S')
    ax[1].set_ylabel('% of initial population')
    ax[1].legend()

    ax[2].plot(100 * td1['R'][:h], label='baseline', linewidth=2)
    ax[2].plot(100 * td2['R'][:h], label='medical capacity', linewidth=2)
    ax[2].set_title('Recovered, R')
    ax[2].set_ylabel('% of initial population')
    ax[2].legend()

    ax[3].plot(100 * td1['D'][:h], label='baseline', linewidth=2)
    ax[3].plot(100 * td2['D'][:h], label='medical capacity', linewidth=2)
    ax[3].set_title('Deaths, D')
    ax[3].set_ylabel('% of initial population')
    ax[3].set_xlabel('weeks')
    ax[3].legend()

    ax[4].plot(100 * (td1['C'] / ss['C'] - 1)[:h], label='baseline', linewidth=2)
    ax[4].plot(100 * (td2['C'] / ss['C'] - 1)[:h], label='medical capacity', linewidth=2)
    ax[4].axhline(0, color='gray', linestyle='--')
    ax[4].set_title('Aggregate Consumption, C')
    ax[4].set_ylabel('% deviation from initial ss')
    ax[4].set_xlabel('weeks')
    ax[4].legend()

    ax[5].plot(100 * (td1['N'] / ss['N'] - 1)[:h], label='baseline', linewidth=2)
    ax[5].plot(100 * (td2['N'] / ss['N'] - 1)[:h], label='medical capacity', linewidth=2)
    ax[5].axhline(0, color='gray', linestyle='--')
    ax[5].set_title('Aggregate Hours, N')
    ax[5].set_ylabel('% deviation from initial ss')
    ax[5].set_xlabel('weeks')
    ax[5].legend()

    ax[6].plot(100 * td1['mortality'][:h], label='baseline', linewidth=2)
    ax[6].plot(100 * td2['mortality'][:h], label='medical capacity', linewidth=2)
    ax[6].set_title('Mortality Rate')
    ax[6].set_ylabel('%')
    ax[6].set_xlabel('weeks')
    ax[6].legend()

    ax[7].plot(100 * td1['ctax'][:h], label='no containment', linewidth=2)
    # ax[7].plot(100 * td3['ctax'][:h], label='optimal containment', linewidth=2)
    ax[7].set_title('Containment Policy')
    ax[7].set_ylabel('%')
    ax[7].set_xlabel('weeks')
    ax[7].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def fig5(h=150, H=250):
    """Discover treatment with constant probability."""
    ss = initial_ss()
    td1 = td_solve(ctax=np.full(H, 0))
    td2 = td_solve(ctax=np.full(H, 0), pr_treat=np.full(H, 1/52))

    fig1, axes = plt.subplots(2, 3, figsize=(.9 * 12, .9 * 8))
    ax = axes.flatten()

    ax[0].plot(100 * td1['I'][:h], label='baseline', linewidth=2)
    ax[0].plot(100 * td2['I'][:h], label='treatment', linewidth=2, linestyle='--')
    ax[0].set_title('Infected, I')
    ax[0].set_ylabel('% of initial population')
    ax[0].legend()

    ax[1].plot(100 * td1['S'][:h], label='baseline', linewidth=2)
    ax[1].plot(100 * td2['S'][:h], label='treatment', linewidth=2, linestyle='--')
    ax[1].set_title('Susceptibles, S')
    ax[1].set_ylabel('% of initial population')
    ax[1].legend()

    ax[2].plot(100 * td1['R'][:h], label='baseline', linewidth=2)
    ax[2].plot(100 * td2['R'][:h], label='treatment', linewidth=2, linestyle='--')
    ax[2].set_title('Recovered, R')
    ax[2].set_ylabel('% of initial population')
    ax[2].legend()

    ax[3].plot(100 * td1['D'][:h], label='baseline', linewidth=2)
    ax[3].plot(100 * td2['D'][:h], label='treatment', linewidth=2, linestyle='--')
    ax[3].set_title('Deaths, D')
    ax[3].set_ylabel('% of initial population')
    ax[3].set_xlabel('weeks')
    ax[3].legend()

    ax[4].plot(100 * (td1['C'] / ss['C'] - 1)[:h], label='baseline', linewidth=2)
    ax[4].plot(100 * (td2['C'] / ss['C'] - 1)[:h], label='treatment', linewidth=2, linestyle='--')
    ax[4].axhline(0, color='gray', linestyle='--')
    ax[4].set_title('Aggregate Consumption, C')
    ax[4].set_ylabel('% deviation from initial ss')
    ax[4].set_xlabel('weeks')
    ax[4].legend()

    ax[5].plot(100 * td1['ctax'][:h], label='no containment', linewidth=2)
    # ax[5].plot(100 * td3['ctax'][:h], label='optimal containment', linewidth=2)
    ax[5].set_title('Containment Policy')
    ax[5].set_ylabel('%')
    ax[5].set_xlabel('weeks')
    ax[5].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def fig6(h=150, H=250):
    """Discover vaccine with constant probability."""
    ss = initial_ss()
    td1 = td_solve(ctax=np.full(H, 0))
    td2 = td_solve(ctax=np.full(H, 0), pr_vacc=np.full(H, 1/52))

    fig1, axes = plt.subplots(2, 3, figsize=(.9 * 12, .9 * 8))
    ax = axes.flatten()

    ax[0].plot(100 * td1['I'][:h], label='baseline', linewidth=2)
    ax[0].plot(100 * td2['I'][:h], label='vaccine', linewidth=2, linestyle='--')
    ax[0].set_title('Infected, I')
    ax[0].set_ylabel('% of initial population')
    ax[0].legend()

    ax[1].plot(100 * td1['S'][:h], label='baseline', linewidth=2)
    ax[1].plot(100 * td2['S'][:h], label='vaccine', linewidth=2, linestyle='--')
    ax[1].set_title('Susceptibles, S')
    ax[1].set_ylabel('% of initial population')
    ax[1].legend()

    ax[2].plot(100 * td1['R'][:h], label='baseline', linewidth=2)
    ax[2].plot(100 * td2['R'][:h], label='vaccine', linewidth=2, linestyle='--')
    ax[2].set_title('Recovered, R')
    ax[2].set_ylabel('% of initial population')
    ax[2].legend()

    ax[3].plot(100 * td1['D'][:h], label='baseline', linewidth=2)
    ax[3].plot(100 * td2['D'][:h], label='vaccine', linewidth=2, linestyle='--')
    ax[3].set_title('Deaths, D')
    ax[3].set_ylabel('% of initial population')
    ax[3].set_xlabel('weeks')
    ax[3].legend()

    ax[4].plot(100 * (td1['C'] / ss['C'] - 1)[:h], label='baseline', linewidth=2)
    ax[4].plot(100 * (td2['C'] / ss['C'] - 1)[:h], label='vaccine', linewidth=2, linestyle='--')
    ax[4].axhline(0, color='gray', linestyle='--')
    ax[4].set_title('Aggregate Consumption, C')
    ax[4].set_ylabel('% deviation from initial ss')
    ax[4].set_xlabel('weeks')
    ax[4].legend()

    ax[5].plot(100 * td1['ctax'][:h], label='no containment', linewidth=2)
    # ax[5].plot(100 * td3['ctax'][:h], label='optimal containment', linewidth=2)
    ax[5].set_title('Containment Policy')
    ax[5].set_ylabel('%')
    ax[5].set_xlabel('weeks')
    ax[5].legend()

    plt.tight_layout()
    plt.show()
    plt.close()
