#include <algorithm>
#include <random>
#include <vector>
#include <array>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <iterator>
#include <fstream>
#include <list>
#include <cassert>
#include <sstream>
#include <future>

// Using default seed, deterministic
std::mt19937_64 gen;

template<class T> struct base_type { typedef T type; };
template<class T> struct base_type<const T&> { typedef T type; };
template<class T> struct base_type<const T> { typedef T type; };
template<class T> struct base_type<T&> { typedef T type; };

template<class ValVector, class ParetoSet>
class Perturber
{
public:
	typedef typename base_type<decltype(*std::begin(ValVector()))>::type Real;

	template<class VA, class VB, class VC, class VD>
	Perturber(size_t n, const VA& traversal_ini_scale, const VB& location_ini_scale,
			const VC& lower_limit, const VD& upper_limit,
			const ParetoSet& archive, const Real& temperature):
		traversals_stat(n),
		locations_stat(n),
		traversal_scale(std::begin(traversal_ini_scale), std::end(traversal_ini_scale)),
		location_scale(std::begin(location_ini_scale), std::end(location_ini_scale)),
		lower_limit(std::begin(lower_limit), std::end(lower_limit)),
		upper_limit(std::begin(upper_limit), std::end(upper_limit)),
		coin(0,1),
		pareto_set(archive),
		T(temperature)
	{}

	bool perturb(size_t direction, Real& val)
	{
		bool is_location = coin(gen);
		Real rad = is_location ? location_scale[direction] : traversal_scale[direction];

		Real from = std::max(val - rad, lower_limit[direction]);
		Real to = std::min(val + rad, upper_limit[direction]);

		// Paper says to use Lagrange distribution, but I find it silly
		// so I am just using uniform distribution...
		val = std::uniform_real_distribution<Real>(from, to)(gen);

		return is_location;
	}

	void add_location_stat(size_t direction, bool accepted)
	{
		LocationStat& dir_stat = locations_stat[direction];
		++dir_stat.total_count;
		if(accepted)
			++dir_stat.taken_count;

		if(dir_stat.total_count >= 20 && can_update_location())
		{
			Real alpha = dir_stat.taken_count / float(dir_stat.total_count);
			Real &sigma = location_scale[direction]; 
			if(alpha > 0.4)
				sigma *= 1.0 + 2.0 * (alpha - 0.4) / 0.6;
			else if(alpha < 0.3)
				sigma /= 1.0 + 2.0 * (0.3 - alpha) / 0.3;

			dir_stat = LocationStat();
		}
	}

	void add_traversal_stat(size_t direction, Real param_step_size, const ValVector& curr, const ValVector& next)
	{
		// Calculate distance taken from the previous point in solution space
		Real dist = 0.0;
		auto cv = std::begin(curr);
		for(auto nv: next) {
			Real dif = nv - *cv;
			dist += dif * dif;
			++cv;
		}
		dist = std::sqrt(dist);

		// Store the size of the step taken in search space and
		// the size of the displacement in solution space, to recalculate
		// the scale of the search random perturbation.
		std::vector<TraversalStat> &dir_stat = traversals_stat[direction];
		dir_stat.push_back({std::abs(param_step_size), dist});

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
	bool can_update_location()
	{
		return pareto_set.size() >= 10 && (T * pareto_set.size() /* TODO: take attainment surface into account */ > 1.0);
	}

	struct TraversalStat {
		Real param_step_size;
		Real obj_dist;

		bool operator<(const TraversalStat& other) const
		{
			return param_step_size < other.param_step_size;
		}
	};

	struct LocationStat {
		LocationStat():
			taken_count(0),
			total_count(0)
		{}

		size_t taken_count;
		size_t total_count;
	};

	std::vector<std::vector<TraversalStat>> traversals_stat;
	std::vector<LocationStat> locations_stat;

	std::vector<Real> traversal_scale;
	std::vector<Real> location_scale;

	std::vector<Real> lower_limit;
	std::vector<Real> upper_limit;

	std::uniform_int_distribution<unsigned short> coin;

	const ParetoSet &pareto_set;
	const Real &T;
};

template<class IterA, class IterB>
bool dominates(IterA a, IterA a_end, IterB b, bool has_smaller=false) {
	do {
		if(*a > *b)
			return false;
		else if(*a < *b)
			has_smaller = true;
		++a;
		++b;
	} while(a != a_end);
	return has_smaller;
};

template<class Vector>
bool dominates(const Vector& a, const Vector& b) {
	return dominates(std::begin(a), std::end(a), std::begin(b));
}

// TODO: use a smart algorithm to maintain this set...
// http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1237166&tag=1
template<class Individual>
class ParetoSet
{
public:
	ParetoSet() = default;

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
		};

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

	size_t size() const
	{
		return pareto_set.size();
	}

	std::list<Individual> get_inner_set()
	{
		return pareto_set;
	}

private:
	std::list<Individual> pareto_set;
};

template<class Vector, class ValVector>
struct Individual {
	Vector x;
	ValVector vals;
};

template<class Vector, class Function, class Real>
auto MOSA(const Vector& x_ini, Function obj_funcs, const Vector& lower_limit, const Vector& upper_limit,
		Real T_ini, Real rt, const Vector& ini_search_radius, size_t Nt, size_t Ne)
	-> decltype(ParetoSet<Individual<Vector, decltype(obj_funcs(x_ini))>>().get_inner_set())
{
	static_assert(std::is_same<decltype(*std::begin(obj_funcs(x_ini))), const Real&>::value,
		"obj_funcs must return an interable of values of type Real");

	typedef decltype(obj_funcs(x_ini)) ValVector;
	typedef Individual<Vector, ValVector> Individual;

	std::uniform_real_distribution<Real> unirand;

	std::vector<size_t> n(x_ini.size(), 0);
	Real T = T_ini;
	Individual current = {x_ini, obj_funcs(x_ini)};

	ParetoSet<Individual> archive(current);
	Perturber<ValVector, decltype(archive)> perturber(x_ini.size(), ini_search_radius, ini_search_radius, lower_limit, upper_limit, archive, T);
	int curr_energy = 0;

	// At each iteration of this loop, temperature is decreased
	for(size_t i = 0; i < Ne; ++i) {

		// Iterations with the same temperature
		for(size_t m = 0; m < Nt; ++m) {

			// Perturb solution one direction at a time
			// (it seems weird, but the algorithm is described as this)
			size_t direction = 0;
			for(auto &e: current.x)
			{
				Real orig_e = e;

				bool is_location = perturber.perturb(direction, e);
				ValVector next_val = obj_funcs(current.x);

				// TODO: Improve energy resolution with attainment surface here...

				int new_energy = archive.try_include(current.x, next_val);

				if(new_energy <= 0) {
					// This move takes us closer to real Pareto set

					// Check if this is a traversal move that needs statistics
					if(!is_location && curr_energy == 0 &&
							(new_energy == 0 || !dominates(next_val, current.vals))) {
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
						if(dominates(current.vals, next_val)) {
							--delta;
						} else if(dominates(next_val, current.vals)) {
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
						current.vals = next_val;
						curr_energy = new_energy;
					} else {
						// Setback, this path was not taken...
						e = orig_e;
					}
				}
				++direction;
			}
		}
		T *= rt;
		//std::cout << i << std::endl;
	}

	return archive.get_inner_set();
}

template<class T>
void print_set(const char* filename, T pareto_set)
{
	std::ofstream file(filename);
	for(const auto &ind: pareto_set) {
		auto ival = std::begin(ind.vals);
		file << *(ival++);
		while(ival != std::end(ind.vals)) {
			file << ' ' << *(ival++);
		}
		file << '\n';
	}
}

namespace ZDT {

typedef std::array<double, 10> Vector;
typedef std::array<double, 2> ValVector;

double ZDT_f1(double x){
	return x;
}

template <class G, class H, class F1=decltype(ZDT_f1)>
void run_ZDT(const char* name, const Vector& start, const Vector& min_limits, const Vector& max_limits, const Vector& search_radius, G g, H h, F1 f1=ZDT_f1)
{
	size_t call_count = 0;

	auto zdt = [&](const Vector& x) {
		ValVector ret;
		ret[0] = f1(x[0]);
		double g_res = g(x);
		ret[1] = g_res * h(ret[0], g_res);

		++call_count;

		return ret;
	};

	std::stringstream fname;
	fname << name << ".dat";
	print_set(fname.str().c_str(),
			MOSA(start, zdt, min_limits, max_limits, 1.0, 0.88, search_radius, 100, 100));
	std::cout << name << ", called " << call_count << " times." << std::endl;
}

};

template<class Vector, class Real>
Vector filled_vector(Real value)
{
	Vector ret;
	std::fill(ret.begin(), ret.end(), value);
	return ret;
}

template<class F>
std::future<void> async(F f)
{
	return std::async(std::launch::async, f);
}

void test_MOSA()
{
	std::vector<std::future<void>> futures;

	//SCH1 test
	futures.push_back(async([]()
	{
		typedef std::array<double, 1> Vector;
		typedef std::array<double, 2> ValVector;
		size_t call_count = 0;

		auto SCH1 = [&](const Vector& x) {
			ValVector ret;
			ret[0] = x[0]*x[0];
			ret[1] = x[0] - 2.0;
			ret[1] *= ret[1];

			++call_count;

			return ret;
		};

		print_set("SCH1.dat", MOSA<Vector>({-4.0}, SCH1, {-10.0}, {10.0}, 1.0, 0.88, {8.0}, 100, 100));
		std::cout << "SCH1, called " << call_count << " times." << std::endl;
	}));

	//SCH2 test
	futures.push_back(async([]()
	{
		typedef std::array<double, 1> Vector;
		typedef std::array<double, 2> ValVector;
		size_t call_count = 0;

		auto SCH2 = [&](const Vector& x) {
			ValVector ret;
			double v = x[0];

			if(v <= 1.0) {
				ret[0] = -v;
			} else if(v <= 3.0) {
				ret[0] = v - 2.0;
			} else if(v <= 4.0) {
				ret[0] = 4.0 - v;
			} else {
				ret[0] = v - 4.0;
			}

			ret[1] = x[0] - 5.0;
			ret[1] *= ret[1];

			++call_count;

			return ret;
		};

		print_set("SCH2.dat", MOSA<Vector>({-4.0}, SCH2, {-5.0}, {10.0}, 1.0, 0.88, {6.0}, 100, 100));
		std::cout << "SCH2, called " << call_count << " times." << std::endl;
	}));

	//FON test
	futures.push_back(async([]()
	{
		typedef std::array<double, 10> Vector;
		typedef std::array<double, 2> ValVector;
		size_t call_count = 0;

		auto FON = [&](const Vector& x) {
			ValVector ret;

			double s1 = 0.0;
			double s2 = 0.0;
			for(double xi: x) {
				double tmp = xi - 1.0 / sqrt(10.0);
				s1 += tmp * tmp;
				tmp = xi + 1.0 / sqrt(10.0);
				s2 += tmp * tmp;
			}
			ret[0] = 1.0 - exp(-s1);
			ret[1] = 1.0 - exp(-s2);

			++call_count;

			return ret;
		};

		Vector start;
		Vector min_limits;
		Vector max_limits;
		Vector search_radius;

		std::fill(start.begin(), start.end(), 1.0);
		std::fill(min_limits.begin(), min_limits.end(), -4.0);
		std::fill(max_limits.begin(), max_limits.end(), 4.0);
		std::fill(search_radius.begin(), search_radius.end(), 6.0);

		print_set("FON.dat", MOSA<Vector>(start, FON, min_limits, max_limits, 1.0, 0.88, search_radius, 100, 100));
		std::cout << "FON, called " << call_count << " times." << std::endl;
	}));

	// KUR test
	futures.push_back(async([]()
	{
		typedef std::array<double, 3> Vector;
		typedef std::array<double, 2> ValVector;
		size_t call_count = 0;

		auto KUR = [&](const Vector& x) {
			ValVector ret;
			ret[0] = 0.0;
			for(int i = 0; i < 2; ++i) {
				ret[0] += -10.0 * exp(-0.2 * sqrt(x[i]*x[i] + x[i+1]*x[i+1]));
			}

			ret[1] = 0.0;
			for(auto xi: x) {
				ret[1] += pow(abs(xi), 0.8) + 5.0 * sin(xi*xi*xi);
			}

			++call_count;

			return ret;
		};

		print_set("KUR.dat", MOSA<Vector>({0.0, 0.0, 0.0}, KUR, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0}, 1.0, 0.88, {5.0, 5.0, 5.0}, 300, 200));
		std::cout << "KUR, called " << call_count << " times." << std::endl;
	}));

	// GTP test
	futures.push_back(async([]()
	{
		typedef std::array<double, 30> Vector;
		typedef std::array<double, 2> ValVector;
		size_t call_count = 0;

		auto GTP = [&](const Vector& x) {
			ValVector ret;
			ret[0] = x[0];
			double g = 2.0;
			double prod = 1.0;
			for(int i = 1; i < 30; ++i) {
				g += x[i]*x[i] / 4000.0;
				prod *= cos(x[i] / sqrt(i+1.0));
			}
			g -= prod;
			ret[1] = g * (1.0 - sqrt(x[0] / g));

			++call_count;

			return ret;
		};

		Vector start;
		Vector min_limits;
		Vector max_limits;
		Vector search_radius;

		std::fill(start.begin(), start.end(), 1.0);

		min_limits[0] = 0.0;
		std::fill(min_limits.begin()+1, min_limits.end(), -5.12);
		max_limits[0] = 1.0;
		std::fill(max_limits.begin()+1, max_limits.end(), 5.12);

		std::fill(search_radius.begin(), search_radius.end(), 5.0);

		print_set("GTP.dat", MOSA<Vector>(start, GTP, min_limits, max_limits, 1.0, 0.92, search_radius, 300, 100));
		std::cout << "GTP, called " << call_count << " times." << std::endl;
	}));

	// ZDT tests
	futures.push_back(async([]()
	{
		std::vector<std::future<void>> futures;

		auto g1 = [](const ZDT::Vector& x) {
			double sum = 0.0;
			for(int i = 1; i < x.size(); ++i) {
				sum += x[i];
			}
			return 1.0 + 9.0 * sum/(x.size() - 1);
		};

		auto g2 = [](const ZDT::Vector& x) {
			double sum = 0.0;
			for(int i = 1; i < x.size(); ++i) {
				sum += x[i]*x[i] - 10.0*cos(4*M_PI*x[i]);
			}
			return 1.0 + 10.0 * (x.size() - 1) + sum;
		};

		auto g3 = [](const ZDT::Vector& x) {
			double sum = 0.0;
			for(int i = 1; i < x.size(); ++i) {
				sum += x[i] / 9.0;
			}
			return 1.0 + 9.0 * pow(sum, 0.25);
		};

		auto h1 = [](double f1, double g) {
			return 1.0 - sqrt(f1 / g);
		};

		auto h2 = [](double f1, double g) {
			return 1.0 - pow(f1 / g, 2.0);
		};

		auto h3 = [](double f1, double g) {
			double r = f1/g;
			return 1.0 - sqrt(r) - r * sin(10.0 * M_PI * f1);
		};

		auto zdt6_f1 = [](double x) {
			return 1.0 - exp(-4.0 * x) * pow(sin(6.0 * M_PI * x), 6.0);
		};

		auto start = filled_vector<ZDT::Vector>(0.5);
		auto min_lim = filled_vector<ZDT::Vector>(0.0);
		auto max_lim = filled_vector<ZDT::Vector>(1.0);
		auto sradius = filled_vector<ZDT::Vector>(0.5);

		futures.push_back(async([&](){ZDT::run_ZDT("ZDT1", start, min_lim, max_lim, sradius, g1, h1);}));
		futures.push_back(async([&](){ZDT::run_ZDT("ZDT2", start, min_lim, max_lim, sradius, g1, h2);}));
		futures.push_back(async([&](){ZDT::run_ZDT("ZDT3", start, min_lim, max_lim, sradius, g1, h3);}));
		futures.push_back(async([&](){ZDT::run_ZDT("ZDT4", start, min_lim, max_lim, sradius, g2, h1);}));
		futures.push_back(async([&](){ZDT::run_ZDT("ZDT6", start, min_lim, max_lim, sradius, g3, h2, zdt6_f1);}));

		for(auto& f: futures) {
			f.wait();
		}
	}));

	// VNT test
	futures.push_back(async([]()
	{
		typedef std::array<double, 2> Vector;
		typedef std::array<double, 3> ValVector;
		size_t call_count = 0;

		auto VNT = [&](const Vector& x) {
			ValVector ret;
			
			double tmp = x[0]*x[0] + x[1]*x[1];
			ret[0] = 0.5 * tmp + sin(tmp);

			ret[1] = pow(3.0*x[0] - 2.0*x[1] + 4.0, 2.0) / 8.0 + pow(x[0] - x[1] + 1.0, 2.0) / 27.0 + 15.0;

			ret[2] = 1.0 / (tmp + 1.0) - 1.1 * exp(-tmp);

			++call_count;

			return ret;
		};

		print_set("VNT.dat", MOSA<Vector>({0.0, 0.0}, VNT, {-3.0, -3.0}, {3.0, 3.0}, 1.0, 0.9, {2.0, 2.0}, 100, 150));
		std::cout << "VNT, called " << call_count << " times." << std::endl;
	}));

	// DTLZ2 test
	futures.push_back(async([]()
	{
		typedef std::array<double, 12> Vector;
		typedef std::array<double, 3> ValVector;
		size_t call_count = 0;

		auto DTLZ2 = [&](const Vector& x) {
			ValVector ret;
			
			double g = 1.0;
			for(int i = 2; i < x.size(); ++i) {
				g += (x[i] - 0.5) * (x[i] - 0.5);
			}

			ret[0] = g * cos(0.5*M_PI*x[0]) * cos(0.5*M_PI*x[1]);
			ret[1] = g * cos(0.5*M_PI*x[0]) * sin(0.5*M_PI*x[1]);
			ret[2] = g * sin(0.5*M_PI*x[0]);

			++call_count;

			return ret;
		};

		print_set("DTLZ2.dat", MOSA(filled_vector<Vector>(0.0), DTLZ2, filled_vector<Vector>(0.0), filled_vector<Vector>(1.0), 1.0, 0.90, filled_vector<Vector>(0.4), 100, 150));
		std::cout << "DTLZ2, called " << call_count << " times." << std::endl;
	}));

	for(auto& f: futures) {
		f.wait();
	}
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

void test_SA()
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

int main()
{
	test_MOSA();
}
