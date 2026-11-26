#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>


__global__ void simulate_paths_kernel(double *output,double initial_price, double interest_rate, double volatility, double time_step, int number_of_paths, int number_of_steps, unsigned long long seed, double diffusion, double drift){
    int path_position = blockIdx.x * blockDim.x + threadIdx.x;

    // Dont allow the number of threads to exceed number of paths planned.
    if (path_position >= number_of_paths) return;

    // Start of this path's row in the flattened array
    int row_start = path_position * (number_of_steps + 1);
    
    // Initial price
    output[row_start] = initial_price;

    // Generate random numbers for this path
    curandState generator;
    curand_init(seed, path_position, 0, &generator);

    double prev_price = initial_price;
    for (int i = 1; i <= number_of_steps; i++){
        // Calculate new price
        double random_num = curand_normal_double(&generator);
        double new_price = prev_price * exp(drift + diffusion * random_num);

        // Store new price
        output[row_start + i] = new_price;
        prev_price = new_price;
    }
        
    return;
}


std::vector<double> simulate_paths(double initial_price, double maturity, int number_of_steps, double interest_rate, double volatility, int number_of_paths, unsigned long long seed){
    // Maturity = number of years till option expires
    double time_step = maturity / number_of_steps;
    int number_of_stock_prices = number_of_paths * (number_of_steps + 1);

    // Creating the matrix as one long row
    std::vector<double> stock_prices(number_of_stock_prices);

    // Price formula variables
    double drift = (interest_rate - 0.5 * volatility * volatility) * time_step;
    double diffusion = volatility * std::sqrt(time_step);
    auto context_setup_start = std::chrono::high_resolution_clock::now();
    cudaFree(0);
    auto kernel_setup_start = std::chrono::high_resolution_clock::now();
    // Set up CUDA kernel
    double *d_output;
    cudaMalloc(&d_output, number_of_stock_prices * sizeof(double));

    int threads_per_block = 256;
    int number_of_blocks = (number_of_paths + threads_per_block - 1) / threads_per_block;
    auto kernel_setup_end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for MCTS kernel setup no context: " << std::chrono::duration<double, std::milli>(kernel_setup_end - kernel_setup_start).count() << " ms" << std::endl;
    std::cout << "Time taken for MCTS context setup: " << std::chrono::duration<double, std::milli>(kernel_setup_start - context_setup_start).count() << " ms" << std::endl;
    auto kernel_run_start = std::chrono::high_resolution_clock::now();
    simulate_paths_kernel<<<number_of_blocks, threads_per_block>>>(
    d_output,
    initial_price,
    interest_rate,
    volatility,
    time_step,
    number_of_paths,
    number_of_steps,
    seed,
    diffusion,
    drift
    );
    
    cudaDeviceSynchronize();
    auto kernerl_run_end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for MCTS kernel run: " << std::chrono::duration<double, std::milli>(kernerl_run_end - kernel_run_start).count() << " ms" << std::endl;
    
    auto copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(stock_prices.data(), d_output, number_of_stock_prices * sizeof(double), cudaMemcpyDeviceToHost);
    auto copy_end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for MCTS copy from device to host: " << std::chrono::duration<double, std::milli>(copy_end - copy_start).count() << " ms" << std::endl;
    cudaFree(d_output);
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

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> stock_prices = simulate_paths(initial_price, maturity, number_of_steps,
                                                      interest_rate, volatility, number_of_paths, seed);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milliseconds = end_time - start_time;
    std::cout << "Time taken for MCTS path simulation: " << elapsed_milliseconds.count() << " ms" << std::endl;

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