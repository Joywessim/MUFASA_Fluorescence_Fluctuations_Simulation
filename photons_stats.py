import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance

def study_stats(data, 
                distributions = {'normal': stats.norm,
                                 'laplace': stats.laplace,
                                 'geometric': stats.geom,
                                 'poisson': stats.poisson,
                                 'binomial': stats.binom},
                plot_distribution = True,
                verbose_distribution = True):
    
    fit_results = {}
    metrics = {}

    # Fit each distribution and calculate AIC, BIC, KL Divergence, and EMD
    for name, distribution in distributions.items():
        if name in ['poisson', 'geometric', 'binomial']:
            continue  # Skip poisson, geometric, and binomial

        params = distribution.fit(data)
        fit_results[name] = params
        log_likelihood = np.sum(np.log(distribution.pdf(data, *params)))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(len(data)) * k - 2 * log_likelihood
        
        # Calculate histogram for KL divergence and EMD
        hist_data, bin_edges = np.histogram(data, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        fitted_pdf = distribution.pdf(bin_centers, *params)
        
        # KL Divergence (adding a small constant to avoid division by zero)
        kl_div = entropy(hist_data + 1e-10, fitted_pdf + 1e-10)
        
        # Earth Mover's Distance (EMD)
        emd = wasserstein_distance(data, distribution.rvs(*params, size=len(data)))
        
        metrics[name] = {'log_likelihood': log_likelihood, 'AIC': aic, 'BIC': bic, 'KL Divergence': kl_div, 'EMD': emd}

    # Manually estimate the Poisson distribution parameter
    lambda_poisson = np.mean(data)
    fit_results['poisson'] = (lambda_poisson,)
    poisson_rvs = stats.poisson.rvs(lambda_poisson, size=len(data))
    log_likelihood_poisson = np.sum(np.log(stats.poisson.pmf(data, lambda_poisson)))
    k_poisson = len(fit_results['poisson'])
    aic_poisson = 2 * k_poisson - 2 * log_likelihood_poisson
    bic_poisson = np.log(len(data)) * k_poisson - 2 * log_likelihood_poisson
    hist_data_poisson, _ = np.histogram(data, bins=30, density=True)
    fitted_pmf_poisson = stats.poisson.pmf(np.arange(len(hist_data_poisson)), lambda_poisson)
    kl_div_poisson = entropy(hist_data_poisson + 1e-10, fitted_pmf_poisson + 1e-10)
    emd_poisson = wasserstein_distance(data, poisson_rvs)
    metrics['poisson'] = {'log_likelihood': log_likelihood_poisson, 'AIC': aic_poisson, 'BIC': bic_poisson, 'KL Divergence': kl_div_poisson, 'EMD': emd_poisson}

    # Manually estimate the Geometric distribution parameter
    p_geom = 1 / (np.mean(data) + 1)
    fit_results['geometric'] = (p_geom,)
    geom_rvs = stats.geom.rvs(p_geom, size=len(data))
    log_likelihood_geom = np.sum(np.log(stats.geom.pmf(data, p_geom)))
    k_geom = len(fit_results['geometric'])
    aic_geom = 2 * k_geom - 2 * log_likelihood_geom
    bic_geom = np.log(len(data)) * k_geom - 2 * log_likelihood_geom
    hist_data_geom, _ = np.histogram(data, bins=30, density=True)
    fitted_pmf_geom = stats.geom.pmf(np.arange(len(hist_data_geom)), p_geom)
    kl_div_geom = entropy(hist_data_geom + 1e-10, fitted_pmf_geom + 1e-10)
    emd_geom = wasserstein_distance(data, geom_rvs)
    metrics['geometric'] = {'log_likelihood': log_likelihood_geom, 'AIC': aic_geom, 'BIC': bic_geom, 'KL Divergence': kl_div_geom, 'EMD': emd_geom}

    # Manually estimate the Binomial distribution parameter
    n = np.max(data)  # Assuming the maximum value in the data corresponds to the number of trials
    p_binom = np.mean(data) / n  # Estimating the probability of success
    fit_results['binomial'] = (n, p_binom)
    binom_rvs = stats.binom.rvs(n, p_binom, size=len(data))
    log_likelihood_binom = np.sum(np.log(stats.binom.pmf(data, n, p_binom)))
    k_binom = 2  # n and p are the two parameters
    aic_binom = 2 * k_binom - 2 * log_likelihood_binom
    bic_binom = np.log(len(data)) * k_binom - 2 * log_likelihood_binom
    hist_data_binom, _ = np.histogram(data, bins=30, density=True)
    fitted_pmf_binom = stats.binom.pmf(np.arange(len(hist_data_binom)), n, p_binom)
    kl_div_binom = entropy(hist_data_binom + 1e-10, fitted_pmf_binom + 1e-10)
    emd_binom = wasserstein_distance(data, binom_rvs)
    metrics['binomial'] = {'log_likelihood': log_likelihood_binom, 'AIC': aic_binom, 'BIC': bic_binom, 'KL Divergence': kl_div_binom, 'EMD': emd_binom}

    if verbose_distribution:
        print("\nComparison Metrics:")
        for name, metric in metrics.items():
            print(f"{name} - Log Likelihood: {metric['log_likelihood']:.2f}, AIC: {metric['AIC']:.2f}, BIC: {metric['BIC']:.2f}, KL Divergence: {metric['KL Divergence']:.2f}, EMD: {metric['EMD']:.2f}")

    if plot_distribution:
        # Plot the data and all the fit distributions
        plt.figure(figsize=(12, 6))
        plt.hist(data, bins=30, density=True, alpha=0.6, color='r', label='Data')

        # Generate a range of values
        x = np.linspace(min(data), max(data), 100)

        # Plot each fitted distribution
        for name, params in fit_results.items():
            if name == 'poisson':
                plt.plot(x, stats.poisson.pmf(np.round(x), *params), label=name)
            elif name == 'geometric':
                plt.plot(x, stats.geom.pmf(np.round(x), *params), label=name)
            elif name == 'binomial':
                plt.plot(x, stats.binom.pmf(np.round(x), *params), label=name)
            else:
                plt.plot(x, distributions[name].pdf(x, *params),alpha=0.4, label=name)

        plt.xlabel('Photon Emissions per Frame')
        plt.ylabel('Density')
        plt.title('Photon Emissions per Frame Distribution')
        plt.legend()
        plt.grid(True)
        plt.show()




























                                                                                
                                                                                
                                                                                
                                                                                
            #             ,@@@/../@@@*              @@@(,.*@@@(                   
            #           @@            @@         @@            @@                 
            #          @,              ,@@@@(&@@@@               @*               
            # #@,#@  ,@@                @&     /@                %@,.             
            # @       #@                @%     .@                &@               
            # @.       &@              @@       (@.             @@                
            #            @@/        *@@           &@@        .@@                  
            #   .  &         #@@@@%                   (@@@@@.                     
                                                                                
            #            @             .@@(     @@(             .@                
            #            @@*        @@@@@@@@@@@@@@@@@@@        &@&                
            #            %@@@*  ,@@@@@@@@@@@@@@@@@@@@@@@@@.  (@@@.                
            #             @@@@@@@@@@@@@@@#       @@@@@@@@@@@@@@@@                 
            #             *@@@@@@@         @@@@@        ,@@@@@@@                  
            #              @@@@@@@@       @@@@@@@       @@@@@@@,                  
            #               #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                    
            #                 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                     
            #                  /@@@@@@@@@@@@@@@@@@@@@@@@@@@                       
            #                    .@@@@@@@@@@@@@@@@@@@@@@@                         
            #                        @@@@@@@@@@@@@@@@&                            
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
