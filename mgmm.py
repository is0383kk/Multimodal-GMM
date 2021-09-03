import numpy as np
from scipy.stats import  wishart, dirichlet 
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ari


############################## Loading a Data Set ##############################
print("Loading a Data Set")
x_nd_1 = np.loadtxt("./data1.txt") # Observation1(Corresponds to x_1 in the graphical model)
x_nd_2 = np.loadtxt("./data2.txt") # Observation2(Corresponds to x_2 in the graphical model)
z_truth_n = np.loadtxt("./true_label.txt") # True label (True z_n)
K = 3 # Number of clusters
D = len(x_nd_1) # Number of data
dim = len(x_nd_1[0]) # Number of dimention
print(f"Number of clusters: {K}"); print(f"Number of data: {len(x_nd_1)}"); print(f"Number of dimention: {len(x_nd_1[0])}")
iteration = 50 # Iteration of gibbssampling
ARI = np.zeros((iteration)) # ARI per iteration


############################## Initializing parameters ##############################
# Please refer to the graphical model in README.
print("Initializing parameters")

# Set hyperparameters
alpha_k = np.repeat(2.0, K) # Hyperparameters for \pi
beta = 1.0; m_d_1 = np.repeat(0.0, dim); m_d_2 = np.repeat(0.0, dim) # Hyperparameters for \mu^A, \mu^B
w_dd_1 = np.identity(dim) * 0.05; w_dd_2 = np.identity(dim) * 0.05 # Hyperparameters for \Lambda^A, \Lambda^B
nu = dim # Hyperparameters for \Lambda^A, \Lambda^B (nu > Number of dimention - 1)

# Initializing \pi
pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten()
alpha_hat_k = np.zeros(K)

# Initializing z
z_nk = np.zeros((D, K)) 
z_nk = np.random.multinomial(n=1, pvals=pi_k, size=D)
_, z_n = np.where(z_nk == 1)

# Initializing unsampled \mu, \Lambda
mu_kd_1 = np.empty((K, dim)); lambda_kdd_1 = np.empty((K, dim, dim))
mu_kd_2 = np.empty((K, dim)); lambda_kdd_2 = np.empty((K, dim, dim))

# Initializing learning parameters
eta_nk = np.zeros((D, K))
beta_hat_k_1 = np.zeros(K) ;beta_hat_k_2 = np.zeros(K)
m_hat_kd_1 = np.zeros((K, dim)); m_hat_kd_2 = np.zeros((K, dim))
w_hat_kdd_1 = np.zeros((K, dim, dim)); w_hat_kdd_2 = np.zeros((K, dim, dim))
nu_hat_k_1 = np.zeros(K); nu_hat_k_2 = np.zeros(K)

# Variables for storing the transition of each parameter
trace_z_in = [z_n.copy()]
trace_mu_ikd_1 = [np.repeat(np.nan, D)]; trace_mu_ikd_2 = [np.repeat(np.nan, D)]
trace_lambda_ikdd_1 = [np.repeat(np.nan, D)]; trace_lambda_ikdd_2 = [np.repeat(np.nan, D)]
trace_beta_ik_1 = [np.repeat(beta, K)]; trace_beta_ik_2 = [np.repeat(beta, K)]
trace_m_ikd_1 = [np.repeat(m_d_1.reshape((1, dim)), K, axis=0)]; trace_m_ikd_2 = [np.repeat(m_d_2.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_1 = [np.repeat(w_dd_1.reshape((1, dim, dim)), K, axis=0)]; trace_w_ikdd_2 = [np.repeat(w_dd_2.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_1 = [np.repeat(nu, K)]; trace_nu_ik_2 = [np.repeat(nu, K)]
trace_pi_ik = [pi_k.copy()]
trace_alpha_ik = [alpha_k.copy()]


############################## Gibbssampling ##############################
print("Gibbssampling")
for i in range(iteration):
    print(f"----------------------Iteration : {i+1}------------------------")
    z_pred_n = [] # Labels estimated by the model
    
    # Process on sampling \mu, \lambda
    for k in range(K):
        # Calculate the parameters of the posterior distribution of \mu
        beta_hat_k_1[k] = np.sum(z_nk[:, k]) + beta; beta_hat_k_2[k] = np.sum(z_nk[:, k]) + beta
        m_hat_kd_1[k] = np.sum(z_nk[:, k] * x_nd_1.T, axis=1); m_hat_kd_2[k] = np.sum(z_nk[:, k] * x_nd_2.T, axis=1)
        m_hat_kd_1[k] += beta * m_d_1; m_hat_kd_2[k] += beta * m_d_2
        m_hat_kd_1[k] /= beta_hat_k_1[k]; m_hat_kd_2[k] /= beta_hat_k_2[k]
        
        # # Calculate the parameters of the posterior distribution of \Lambda
        tmp_w_dd_1 = np.dot((z_nk[:, k] * x_nd_1.T), x_nd_1); tmp_w_dd_2 = np.dot((z_nk[:, k] * x_nd_2.T), x_nd_2)
        tmp_w_dd_1 += beta * np.dot(m_d_1.reshape(dim, 1), m_d_1.reshape(1, dim)); tmp_w_dd_2 += beta * np.dot(m_d_2.reshape(dim, 1), m_d_2.reshape(1, dim))
        tmp_w_dd_1 -= beta_hat_k_1[k] * np.dot(m_hat_kd_1[k].reshape(dim, 1), m_hat_kd_1[k].reshape(1, dim))
        tmp_w_dd_2 -= beta_hat_k_2[k] * np.dot(m_hat_kd_2[k].reshape(dim, 1), m_hat_kd_2[k].reshape(1, dim))
        tmp_w_dd_1 += np.linalg.inv(w_dd_1); tmp_w_dd_2 += np.linalg.inv(w_dd_2)
        w_hat_kdd_1[k] = np.linalg.inv(tmp_w_dd_1); w_hat_kdd_2[k] = np.linalg.inv(tmp_w_dd_2)
        nu_hat_k_1[k] = np.sum(z_nk[:, k]) + nu
        nu_hat_k_2[k] = np.sum(z_nk[:, k]) + nu
        
        # Sampling \Lambda
        lambda_kdd_1[k] = wishart.rvs(size=1, df=nu_hat_k_1[k], scale=w_hat_kdd_1[k])
        lambda_kdd_2[k] = wishart.rvs(size=1, df=nu_hat_k_2[k], scale=w_hat_kdd_2[k])
        
        # Sampling \mu
        mu_kd_1[k] = np.random.multivariate_normal(
            mean=m_hat_kd_1[k], cov=np.linalg.inv(beta_hat_k_1[k] * lambda_kdd_1[k]), size=1
        ).flatten()
        mu_kd_2[k] = np.random.multivariate_normal(
            mean=m_hat_kd_2[k], cov=np.linalg.inv(beta_hat_k_2[k] * lambda_kdd_2[k]), size=1
        ).flatten()
    
    # Process on sampling z
    # Calculate the parameters of the posterior distribution of z
    for k in range(K):
        tmp_eta_n = np.diag(
            -0.5 * (x_nd_1 - mu_kd_1[k]).dot(lambda_kdd_1[k]).dot((x_nd_1 - mu_kd_1[k]).T)
        ).copy() 
        tmp_eta_n += np.diag(
            -0.5 * (x_nd_2 - mu_kd_2[k]).dot(lambda_kdd_2[k]).dot((x_nd_2 - mu_kd_2[k]).T)
        ).copy() 
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_1[k]) + 1e-7)
        tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_2[k]) + 1e-7)
        tmp_eta_n += np.log(pi_k[k] + 1e-7) 
        eta_nk[:, k] = np.exp(tmp_eta_n)
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) # Normalization
    
    # Sampling z
    for d in range(D):
        z_nk[d] = np.random.multinomial(n=1, pvals=eta_nk[d], size=1).flatten()
        z_pred_n.append(np.argmax(z_nk[d])) # Append labels estimated by the model

    # Process on sampling \pi
    # Calculate the parameters of the posterior distribution of \pi
    alpha_hat_k = np.sum(z_nk, axis=0) + alpha_k
    
    # Sampling \pi
    pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
    
    # Calculate ARI
    ARI[i] = np.round(ari(z_truth_n, z_pred_n), 3)
    print(f"ARI:{ARI[i]}")

    # Stores the value of a parameter for each iteration
    _, z_n = np.where(z_nk == 1)
    trace_z_in.append(z_n.copy())
    trace_mu_ikd_1.append(mu_kd_1.copy())
    trace_lambda_ikdd_1.append(lambda_kdd_1.copy())
    trace_beta_ik_1.append(beta_hat_k_1.copy())
    trace_m_ikd_1.append(m_hat_kd_1.copy())
    trace_w_ikdd_1.append(w_hat_kdd_1.copy())
    trace_nu_ik_1.append(nu_hat_k_1.copy())
    trace_pi_ik.append(pi_k.copy())
    trace_alpha_ik.append(alpha_hat_k.copy())


# plot ARI
plt.plot(range(0,iteration), ARI, marker="None")
plt.xlabel('iteration')
plt.ylabel('ARI')
#plt.savefig("./image/ari.png")
plt.show()
plt.close()