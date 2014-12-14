#include <algorithm>
#include <random>
#include <vector>
#include <array>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <iterator>
#include <list>

// Using default seed, deterministic
std::mt19937_64 gen;

template<class ValVector>
class Perturber
{
public:
	typedef decltype(*std::begin(ValVector())) Real;

	template<class ScaleVectorA, class ScaleVectorB>
	Perturber(size_t n, const ScaleVectorA& traversal_ini_scale, const ScaleVectorB& location_ini_scale):
		traversals_stat(n),
		traversal_scale(std::begin(traversal_ini_scale), std::end(traversal_ini_scale)),
		location_scale(std::begin(location_ini_scale), std::end(location_ini_scale)),
		coin(0,1)
	{}

	bool perturb(size_t direction, Real& val)
	{
		bool is_location = coin(gen);
		Real rad = is_location ? location_scale[direction] : traversal_scale[direction];

		// Paper says to use Lagrange distribution, but I find it silly
		// so I am just using uniform distribution...
		val += std::uniform_real_distribution<Real>(-rad, rad)(gen);

		return is_location;
	}

	void add_location_stat(size_t direction, bool accepted)
	{
		LocationStat& dir_stat = locations_stat[direction];
		dir_stat.taken.set(dir_stat.count++, accepted);

		if(dir_stat.count == dir_stat.taken.size())
		{
			Real alpha = dir_stat.taken.count() / float(dir_stat.taken.size());
			Real &sigma = location_scale[direction]; 
			if(alpha > 0.4)
				sigma *= 1.0 + 2.0 * (alpha - 0.4) / 0.6;
			else if(alpha < 0.3)
				sigma /= 1.0 + 2.0 * (0.3 - alpha) / 0.3;

			dir_stat.count = 0;
		}
	}

	void add_traversal_stat(size_t direction, Real param_step_size, const ValVector& curr, const ValVector& next)
	{
		// Calculate distance taken from the previous point in solution space
		Real dist = 0.0;
		auto cv = std::begin(curr);
		for(nv: next) {
			Real dif = nv - *cv;
			dist += dif * dif;
			++cv;
		}
		dist = std::sqrt(dist);

		// Store the size of the step taken in search space and
		// the size of the displacement in solution space, to recalculate
		// the scale of the search random perturbation.
		std::vector<TraversalStat> &dir_stat = traversals_stat[direction];
		dir_stat.push_back({std::abs(param_step_size), obj_dist});

		// If we have enough stored statistics, calculate the new scale
		if(dir_stat.size() == 51) {
			// Sort according to search step size
			std::sort(dir_stat.begin(), dir_stat.end());

			size_t splits[3];
			Real avg[3] = {0.0};

			// Split the sorted statistics in 3 equal parts
			std::fill_n(splits, 3, dir_stat.size() / 3);

			// Find the group that gave most distant walks
			auto iter = dir_stat.begin();
			decltype(iter) begins[3];
			int max = 0;
			for(int j = 0; j < 3; ++j) {
				begins[j] = iter;
				auto end = iter + splits[j];
				for(; iter < end; ++iter) {
					avg[j] += iter->obj_dist;
				}
				avg[j] /= splits[j];
				if(avg[j] > avg[max])
					max = j;
			}

			// Find the mean step size used in the group as basis
			// for the scaler in this direction
			iter = begins[max];
			auto end = iter + splits[max];
			Real new_scale = 0.0;
			for(; iter < end; ++iter) {
				new_scale += iter->param_step_size;
			}
			new_scale = 2.0 * new_scale / splits[max];
			traversal_scale[direction] = new_scale;

			dir_stat.clear();
		}
		assert(dir_stat.size() < 51);
	}

private:
	struct TraversalStat {
		Real param_step_size;
		Real obj_dist;

		bool operator<(const TraversalStat& other) 
		{
			return param_step_size < other.param_step_size;
		}
	};

	struct LocationStat {
		LocationStat():
			count(0)
		{}

		std::bitset<20> taken;
		unsigned char count;
	};

	std::vector<std::vector<TraversalStat>> traversals_stat;
	std::vector<LocationStat> locations_stat;
	std::vector<Real> traversal_scale;
	std::vector<Real> location_scale;

	std::uniform_int_distribution<unsigned short> coin;
};

template<class ValVector>
bool dominates(const ValVector& a, const ValVector& b) {
	auto av = std::begin(a);
	auto bv = std::begin(b);
	do {
		if(*bv < *av)
			return false;
		++av;
		++bv;
	} while(av != std::end(a));
	return true;
}

// TODO: use a smart algorithm to maintain this set...
// http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1237166&tag=1
template<class Individual>
class ParetoSet
{
public:
	ParetoSet(const Individual& ind) {
		pareto_set.push_back(ind);
	}

	typedef decltype(Individual::x) Vector;
	typedef decltype(Individual::vals) ValVector;

	int try_include(const Vector& x, const ValVector& vals) {
		int dominated_count = 0;
		auto p = pareto_set.begin();
		do {
			bool dominated_by_p = true;
			bool dominating_p = false;
			bool can_dominate = true;
			auto iv = std::begin(vals);
			auto pv = std::begin(p->vals);

			do {
				if(*iv < *pv) {
					dominated_by_p = false;
					if(can_dominate) {
						dominating_p = true;
						while(++iv != std::end(vals)) {
							++pv;
							if(*iv > *pv) {
								dominating_p = false;
								break;
							}
						}
					}
					break;
				} else if(*pv < *iv) {
					can_dominate = false;
				}
				++pv;
				++iv;
			} while(iv != std::end(vals));

			if(dominated_by_p) {
				++dominated_count;
				break;
			} else if(dominating_p) {
				*(p++) = {x, vals};
				--dominated_count;
				break;
			}
			++p;
		} while(p != pareto_set.end());

		auto dominates = [](const ValVector& a, const ValVector& b) {
			auto av = std::begin(a);
			auto bv = std::begin(b);
			do {
				if(*bv < *av)
					return false;
				++av;
				++bv;
			} while(av != std::end(a));
			return true;
		}

		if(dominated_count > 0) {
			while(p != pareto_set.end()) {
				if(dominates(p->vals, vals)) {
					++dominated_count;
				}
				++p;
			}
		} else if(dominated_count < 0) {
			while(p != pareto_set.end()) {
				if(dominates(vals, p->vals)) {
					auto tmp = p++;
					pareto_set.erase(tmp);
					--dominated_count;
				} else {
					++p;
				}
			}
		} else {
			pareto_set.push_back({x, vals});
		}

		return dominated_count;
	}

	size_t size()
	{
		return pareto_set.size();
	}

private:
	std::list<Individual> pareto_set;
};

template<class Vector, class Function, class ValidationFunction, class Real>
Vector MOSA(const Vector& x_ini, Function obj_funcs, ValidationFunction is_valid,
		Real T_ini, Real rt, const Vector& ini_search_radius, size_t Ns, size_t Nt, size_t Ne)
{
	static_assert(std::is_same<decltype(*std::begin(obj_funcs(x_ini))), Real>::value,
		"obj_funcs must return an interable of values of type Real");
	typedef decltype(obj_funcs(x_ini)) ValVector;

	struct Individual {
		Vector x;
		ValVector vals;
	};

	std::uniform_real_distribution<Real> unirand;

	std::vector<size_t> n(x_ini.size(), 0);
	Real T = T_ini;
	Individual current = {x_ini, obj_funcs(x_ini)};

	ParetoSet<Individual> archive(current);
	Perturber<ValVector> perturber(x_ini.size(), ini_search_radius, ini_search_radius);
	int curr_energy = 0;

	// At each iteration of this loop, temperature is decreased
	for(size_t i = 0; i < Ne; ++i) {

		// Iterations with the same temperature
		for(size_t m = 0; m < Nt; ++m) {

			// Perturb solution one direction at a time
			// (it seems weird, but the algorithm is described as this)
			size_t direction = 0;
			auto rad = std::begin(search_radius);
			for(auto &e: current.x)
			{
				Real orig_e = e;

				bool is_location = perturber.perturb(direction, e);
				ValVector next_val = obj_funcs(current.x);

				// TODO: Improve energy resolution with attainment surface here...

				int new_energy = archive.try_include(current.x, next_val);

				if(new_energy <= 0 && is_valid(current.x)) {
					// This move takes us closer to real Pareto set

					// Check if this is a traversal move that needs statistics
					if(!is_location && curr_energy == 0 &&
							(new_energy == 0 || !dominates(new_val, current.vals))) {
							// Muttually non-dominated solutions, report statistics
							perturber.add_traversal_stat(direction, e - orig_e, current.vals, next_val);
					}

					// Use this move to continue search
					current.vals = next_val;
					curr_energy = 0;
					// TODO: compute new sampling from attainment surface, since pareto set changed...

				} else {
					// This move is yields a dominated solution: use Metropole test,
					// may be chosen according to Boltzmann distribution.

					int delta = curr_energy - new_energy;
					if(curr_energy > 0) {
						// Current solution is not in pareto set, so new and current
						// solutions must be tested against each another
						if(dominates(current.vals, new_val)) {
							--delta;
						} else if(dominates(new_val, current.vals)) {
							++delta;
						} else if(!is_location) {
							// Both are mutually non-dominant, store statistics
							perturber.add_traversal_stat(direction, e - orig_e, current.vals, next_val);
						}
					}

					bool taken = true;
					if(delta < 0) {
						if(unirand(gen) >= std::exp(Real(delta) / archive.size() / T)) {
							taken = false;
						}

						if(is_location) {
							// Store statistics for recalculate scale
							perturber.add_location_stat(direction, taken);
						}
					}

					if(taken) {
						// This solution was taken to be explored...
						current.val = next_val;
						curr_energy = new_energy;
					} else {
						// Setback, this path was not taken...
						e = orig_e;
					}
				}
				++rad;
				++direction;
			}
		}
		T *= rt;
	}

	return best.x;
}

template<class Vector, class Function, class ValidationFunction, class Real>
Vector SA(const Vector& x_ini, Function obj_func, ValidationFunction is_valid,
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
		// At each iteration of this loop, search radius is adjusted
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

	auto result = SA(X, obj_func, is_valid, 1000.0, 0.85, {1.0, 1.0}, {2.0, 2.0}, 20, 100, 4, 1e-8);

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
