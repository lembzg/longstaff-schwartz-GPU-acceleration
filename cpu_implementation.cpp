#include <vector>
#include <random>
#include <cmath>
#include <iostream>

std::vector<double> simulate_paths(double initial_price, double maturity, int number_of_steps, double interest_rate, double volatility, int number_of_paths, unsigned long long seed){
    // Maturity = number of years till option expires
    double time_step = maturity / number_of_steps;
    int number_of_stock_prices = number_of_paths * (number_of_steps + 1);

    // Creating the matrix as one long row
    std::vector<double> stock_prices(number_of_stock_prices);

    // Initialise initial price
    for (int path = 0; path < number_of_paths; path ++){
        stock_prices[path * (number_of_steps + 1)] = initial_price;
    }
    

    // Filling up prices in each path

    // Create dice
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    double drift = (interest_rate - 0.5 * volatility * volatility) * time_step;
    double diffusion = volatility * std::sqrt(time_step);
    
    // Fill price
    for (int path = 0; path < number_of_paths; path ++){
        for (int date = 1; date <= number_of_steps; date ++){
            double random_num = standard_normal(generator);
            double prev_price = stock_prices[path * (number_of_steps + 1) + date - 1];

            // store new price
            stock_prices[path * (number_of_steps + 1) + date] = prev_price * std::exp(drift + diffusion * random_num);

        }
    }

    return stock_prices;
}

int main()
{
    double initial_price = 36.0;
    double strike = 40.0;
    double maturity = 1.0;
    int number_of_steps = 50;
    double interest_rate = 0.06;
    double volatility = 0.2;
    int number_of_paths = 1000000;
    unsigned long long seed = 42;

    std::vector<double> stock_prices = simulate_paths(initial_price, maturity, number_of_steps,
                                                      interest_rate, volatility, number_of_paths, seed);

    // Check 1: average final price should be close to initial_price * e^(r * maturity)
    double total_final_price = 0.0;
    for (int path = 0; path < number_of_paths; path++) {
        double final_price = stock_prices[path * (number_of_steps + 1) + number_of_steps];
        total_final_price += final_price;
    }
    double average_final_price = total_final_price / number_of_paths;
    double expected_final_price = initial_price * std::exp(interest_rate * maturity);

    std::cout << "Average final price:  " << average_final_price << std::endl;
    std::cout << "Expected:             " << expected_final_price << std::endl;

    // Check 2: European put price should be close to Black-Scholes (3.844)
    double total_put_payoff = 0.0;
    for (int path = 0; path < number_of_paths; path++) {
        double final_price = stock_prices[path * (number_of_steps + 1) + number_of_steps];
        if (strike > final_price) {
            total_put_payoff += strike - final_price;
        }
    }
    double european_put_price = std::exp(-interest_rate * maturity) * total_put_payoff / number_of_paths;

    std::cout << "European put price:   " << european_put_price << std::endl;
    std::cout << "Black-Scholes:        3.844" << std::endl;

    return 0;
}