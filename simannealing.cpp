#include <algorithm>
#include <random>
#include <vector>
#include <array>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <iterator>

// Using default seed, deterministic
std::mt19937_64 gen;

template<class Vector, class Function, class ValidationFunction, class Real>
Vector SE(const Vector& x_ini, Function obj_func, ValidationFunction is_valid,
		Real T_ini, Real rt, Vector search_radius, const Vector search_radius_update_factor,
		size_t Ns, size_t Nt, size_t Ne, Real tol)
{
	static_assert(std::is_same<decltype(obj_func(x_ini)), Real>::value,
		"obj_func must return a value of type Real");

	struct Individual {
		Vector x;
		Real val;
	};

	std::vector<size_t> n(x_ini.size(), 0);
	Real T = T_ini;
	Individual best = {x_ini, obj_func(x_ini)};
	Individual current;

	std::vector<Real> prev_vals(Ne+1, best.val);
	size_t i = 0;

	// At each iteration of this loop, temperature is decreased
	do {
		current = best;

		// Iterations with the same temperature
		// At each iteration of this loop, search radius is reduced
		for(size_t m = 0; m < Nt; ++m) {

			// Iterations with the same search radius
			for(size_t j = 0; j < Ns; ++j) {

				// Perturb solution one direction at a time
				// (it seems weird, but the algorithm is described as this)
				auto rad = std::begin(search_radius);
				auto dir_taken_count = std::begin(n);
				for(auto &e: current.x)
				{
					Real orig_e = e;

					e += std::uniform_real_distribution<Real>(-(*rad), *rad)(gen);

					Real next_val = obj_func(current.x);

					if(next_val <= current.val && is_valid(current.x)) {
						current.val = next_val;
						++(*dir_taken_count);
						if(current.val < best.val) {
							best = current;
						}
					} else {
						// Metropole move, may be chosen according to Boltzmann distribution
						Real p = std::exp((current.val - next_val) / T);
						if(std::uniform_real_distribution<Real>()(gen) < p) {
							current.val = next_val;
							++(*dir_taken_count);
						} else {
							// Setback, this path was not taken...
							e = orig_e;
						}
					}
					++rad;
					++dir_taken_count;
				}
			}

			// Update search radius
			auto dir_taken_count = std::begin(n);
			auto c = std::begin(search_radius_update_factor);
			for(auto &r: search_radius) {
				if(*dir_taken_count > 6 * (Ns / 10))
					r *= (1.0 + (*c) * (float(*dir_taken_count)/Ns - 0.6) / 0.4);
				else if(*dir_taken_count < 4 * (Ns / 10))
					r /= 1.0 + (*c) * (0.4 - float(*dir_taken_count) / Ns) / 0.4;
				++dir_taken_count;
				++c;
			}

			n.assign(n.size(), 0);
		}

		T *= rt;
		prev_vals[i++] = current.val;
		if(i > Ne)
			i = 0;

	} while(std::any_of(prev_vals.begin(), prev_vals.end(),
			[=](Real val) {
				return std::abs(current.val - val) > tol;
			})
		|| current.val - best.val > tol);

	return best.x;
}

int main()
{
	std::array<double, 2> X = {0.5, 0.5};
	size_t call_count = 0;

	auto obj_func = [&](std::array<double, 2> p){
		++call_count;
		//return p[0]*p[0] - 3.0*p[0]*p[1] + 4.0*p[1]*p[1] + p[0] - p[1];
		return 100.0 * std::pow(p[0] - p[1]*p[1], 2) - std::pow(1.0 - p[0], 2);
	};
	auto is_valid = [](std::array<double, 2> p) { return p[0] > -1.0 && p[0] < 1.0 && p[1] > -1.0 && p[1] < 1.0; };

	auto result = SE(X, obj_func, is_valid, 1000.0, 0.85, {1.0, 1.0}, {2.0, 2.0}, 20, 100, 4, 1e-8);

	auto print = [&](std::array<double, 2> p) {
		for(auto v: p) {
			std::cout << v << ' ';
		}
		std::cout << obj_func(p) << '\n';
	};
	print(X);
	print(result);
	std::cout << "Objective function calls: " << call_count << std::endl;
}
