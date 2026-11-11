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

// Helper to find determinant
double determinant(double a1, double b1, double c1,
                   double a2, double b2, double c2,
                   double a3, double b3, double c3){
    return a1 * (b2 * c3 - c2 * b3)
         - b1 * (a2 * c3 - c2 * a3)
         + c1 * (a2 * b3 - b2 * a3);
}

double price_american_put(const std::vector<double>& stock_prices, int number_of_paths, int number_of_steps, double strike, double interest_rate, double time_step){
    std::vector<double> cash_flow(number_of_paths);
    std::vector<int> exercise_date(number_of_paths);

    // Initialise cash flow to expiry date
    for (int path = 0; path < number_of_paths; path ++){
        double profit = strike - stock_prices[path * (number_of_steps + 1) + number_of_steps];
        exercise_date[path] = number_of_steps;
        if (profit > 0){
            cash_flow[path] = profit;
        }
        else {
            cash_flow[path] = 0;
        }
    }

    // Go backwards though dates for every path
    for (int date = number_of_steps - 1; date >= 1; date--){
        // Sums needed for best fit
        double count = 0, sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
        double sum_y = 0, sum_xy = 0, sum_x2y = 0;

        for (int path = 0; path < number_of_paths; path ++){
            // Look up stock price for this path on this specific date
            double price = stock_prices[path * (number_of_steps + 1) + date];

            // Only make decision if stock is below strike
            if (price < strike){
                double X = price / strike;
                // What this path pays if the person keeps waiting
                double Y = cash_flow[path] * std::exp(-interest_rate * (exercise_date[path] - date) * time_step);
                
                count += 1;
                sum_x += X;
                sum_x2 += X * X;
                sum_x3 += X * X * X;
                sum_x4 += X * X * X * X;
                sum_y += Y;
                sum_xy += X * Y;
                sum_x2y += X * X * Y;
            }
        }
        
        // Getting constants for the curve
        double D = determinant(count,  sum_x,  sum_x2, sum_x,  sum_x2, sum_x3, sum_x2, sum_x3, sum_x4);
        double D_a = determinant(sum_y,   sum_x,  sum_x2, sum_xy,  sum_x2, sum_x3, sum_x2y, sum_x3, sum_x4);
        double D_b = determinant(count,  sum_y,   sum_x2, sum_x,  sum_xy,  sum_x3, sum_x2, sum_x2y, sum_x4);
        double D_c = determinant(count,  sum_x,  sum_y, sum_x,  sum_x2, sum_xy, sum_x2, sum_x3, sum_x2y);

        double a = D_a / D;
        double b = D_b / D;
        double c = D_c / D;
        
        // Decide whether to exercise today
        for (int path = 0; path < number_of_paths; path ++){
            double price = stock_prices[path * (number_of_steps + 1) + date];

            if (price < strike){
                // What you get if you exercise now
                double exercise_value = strike - price;

                // Worth based on curve
                double X = price / strike;
                double continuation_value = a + b * X + c * X * X;

                // Overide path record if its more worth to exercise now
                if (exercise_value > continuation_value){
                    cash_flow[path] = exercise_value;
                    exercise_date[path] = date;
                }
            }
        }
    }

    // Calculating the output for each day
    double total = 0.0;
    for (int path = 0; path < number_of_paths; path ++){
        total += cash_flow[path] * std::exp(-interest_rate * exercise_date[path] * time_step);
    }
    
    // Average output for the day
    return total / number_of_paths;
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

    double time_step = maturity / number_of_steps;
    double american_put_price = price_american_put(stock_prices, number_of_paths, number_of_steps,
                                                   strike, interest_rate, time_step);

    std::cout << "American put price:   " << american_put_price << std::endl;
    std::cout << "Paper's value:        4.478" << std::endl;

    return 0;
}