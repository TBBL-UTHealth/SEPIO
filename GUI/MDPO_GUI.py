"""
Unified MDPO application for optimization and assessment.

Provides:
- `OptimizerController` and `OptimizationWorker` for threaded optimization with progress signals.
- Adapters to connect GUI data to compute functions (find_snr, limit correction, transforms).
- A tabbed Tk/ttk interface for project setup, optimization, assessment, SEPIO, and plotting.
"""
from __future__ import annotations
import os, sys, pickle, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
import datetime
import numpy as np
from typing import Optional
from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np

try:
	from PyQt5.QtCore import QObject, QThread, pyqtSignal
	_HAS_QT = True
except Exception:
	# Fallback if PyQt5 is not available in analysis environments
	_HAS_QT = False
	class _DummySignal:
		def connect(self, *args, **kwargs):
			return None
		def emit(self, *args, **kwargs):
			return None
	class QObject: pass
	class QThread:  # minimal shim
		def start(self):
			pass
		def isRunning(self):
			return False
	def pyqtSignal(*args, **kwargs):
		return _DummySignal()


class OptimizationWorker(QThread):
	"""
	Runs a selected optimization method in a background thread.
	Emits progress updates and a final result for the GUI to consume.
	"""

	progressChanged = pyqtSignal(int)
	statusChanged = pyqtSignal(str)
	bestFitnessChanged = pyqtSignal(float)
	bestSolutionChanged = pyqtSignal(object)
	finishedWithResult = pyqtSignal(dict)

	def __init__(self,
				 method: str,
				 data_ctx: Dict[str, Any],
				 params: Dict[str, Any],
				 cancel_flag: Optional[Dict[str, bool]] = None):
		super().__init__()
		self.method = method
		self.data_ctx = data_ctx
		self.params = params
		self.cancel_flag = cancel_flag or {"cancel": False}

	def run(self):
		try:
			self.statusChanged.emit(f"Starting {self.method} optimization...")
			# Pre-correct initial entries using limit correction, if available
			try:
				if "initial_population" in self.data_ctx and self.data_ctx["initial_population"] is not None:
					self.data_ctx["initial_population"] = self._limit_correction(np.asarray(self.data_ctx["initial_population"], dtype=float), (3, 2))
				if "initial_solution" in self.data_ctx and self.data_ctx["initial_solution"] is not None:
					self.data_ctx["initial_solution"] = self._limit_correction(np.asarray(self.data_ctx["initial_solution"], dtype=float), (3, 2))
			except Exception:
				# If limiter unavailable or correction fails, proceed without blocking
				pass
			result = self._dispatch()
			self.finishedWithResult.emit(result)
			self.statusChanged.emit("Optimization finished.")
		except Exception as e:
			self.finishedWithResult.emit({"error": str(e)})
			self.statusChanged.emit(f"Error: {e}")

	def _dispatch(self) -> Dict[str, Any]:
		method = self.method.lower()
		if method == "genetic":
			return self._run_genetic()
		if method in ("anneal", "simulated_annealing"):
			return self._run_simulated_annealing()
		if method == "gradient":
			return self._run_gradient_descent()
		if method in ("multianneal",):
			return self._run_multi_anneal()
		if method in ("msgd", "branchbound"):
			return self._run_branch_bound()
		if method == "brute":
			return self._run_brute_force()
		raise ValueError(f"Unknown optimization method: {self.method}")

	# --- Shared utility placeholders (to be wired to existing implementations) ---
	def _find_snr(self, devpos_flat: np.ndarray) -> float:
		finder: Callable[[np.ndarray], float] = self.data_ctx.get("find_snr")
		if finder is None:
			raise RuntimeError("find_snr function not provided in data_ctx")
		# Apply limit correction prior to scoring to ensure constraints
		try:
			limiter: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]] = self.data_ctx.get("limit_correction")
			if limiter is not None and devpos_flat is not None:
				devpos_flat = limiter(np.asarray(devpos_flat, dtype=float), (3, 2))
		except Exception:
			pass
		return finder(devpos_flat)

	def _limit_correction(self, population: np.ndarray, coefficient: Tuple[int, int]) -> np.ndarray:
		limiter: Callable[[np.ndarray, Tuple[int, int]], np.ndarray] = self.data_ctx.get("limit_correction")
		if limiter is None:
			return population
		return limiter(population, coefficient)

	def _check_cancel(self) -> bool:
		return bool(self.cancel_flag.get("cancel"))

	# --- Method stubs ---
	def _run_genetic(self) -> Dict[str, Any]:
		"""
		Genetic algorithm stub.
		Expects data_ctx to supply: initial_population (np.ndarray), and find_snr function.
		Params should include: num_generations, sol_per_pop, num_parents_mating, ss_gens.
		"""
		init_pop = self.data_ctx.get("initial_population")
		if init_pop is None:
			raise RuntimeError("initial_population missing for genetic method")

		num_generations = int(self.params.get("num_generations", 50))
		ss_gens = int(self.params.get("ss_gens", 3))

		population = np.copy(init_pop)
		best_solution = None
		best_fitness = -np.inf
		fitness_history = []

		for gen in range(num_generations):
			if self._check_cancel():
				return {"canceled": True, "generation": gen, "best": best_solution, "fitness": best_fitness}

			# Evaluate fitness
			fitnesses = np.array([self._find_snr(chrom) for chrom in population])
			fitness_history.append((float(np.max(fitnesses)), float(np.mean(fitnesses)), float(np.std(fitnesses))))

			# Track best
			idx_best = int(np.argmax(fitnesses))
			if fitnesses[idx_best] > best_fitness:
				best_fitness = float(fitnesses[idx_best])
				best_solution = np.copy(population[idx_best])
				self.bestFitnessChanged.emit(best_fitness)
				self.bestSolutionChanged.emit(best_solution)

			# Simple selection + mutation placeholder (to be replaced with real GA ops)
			top_k = min(self.params.get("num_parents_mating", 10), population.shape[0])
			parents = population[np.argsort(-fitnesses)[:top_k]]
			# Refill population by noisy copies
			noise_scale = 0.05
			offspring = parents + np.random.normal(0, noise_scale, size=parents.shape)
			# If steady state detected, adjust noise
			if len(fitness_history) > ss_gens and len({round(f[0], 6) for f in fitness_history[-ss_gens:]}) == 1:
				offspring += np.random.normal(0, noise_scale * 0.5, size=offspring.shape)

			# Limit correction
			offspring = self._limit_correction(offspring, coefficient=(3, 2))

			# Rebuild next generation
			repeats = max(1, population.shape[0] // offspring.shape[0])
			population = np.vstack([offspring for _ in range(repeats)])[:init_pop.shape[0]]

			self.progressChanged.emit(int((gen + 1) * 100 / num_generations))

		return {
			"method": "genetic",
			"best_solution": best_solution,
			"best_fitness": best_fitness,
			"fitness_history": fitness_history,
		}

	def _run_simulated_annealing(self) -> Dict[str, Any]:
		initial_solution = self.data_ctx.get("initial_solution")
		bounds = self.data_ctx.get("bounds")  # (lower_bounds, upper_bounds)
		if initial_solution is None or bounds is None:
			raise RuntimeError("initial_solution or bounds missing for annealing")

		lower_bounds, upper_bounds = bounds
		n_iterations = int(self.params.get("anneal_iterations", 40))
		temp = float(self.params.get("anneal_itemp", 100.0))
		min_temp = float(self.params.get("anneal_ftemp", 1e-3))
		cooling_rate = float(self.params.get("anneal_cooling_rate", 0.85))
		cart_step = float(self.params.get("anneal_cart_step", 15.0))
		rot_step = float(self.params.get("anneal_rot_step", np.pi/3))

		current = np.copy(initial_solution).flatten()
		current_f = self._find_snr(current)
		best = np.copy(current)
		best_f = current_f

		history = []
		cycle = 0
		while temp > min_temp:
			cycle += 1
			for it in range(n_iterations):
				if self._check_cancel():
					return {"canceled": True, "cycle": cycle, "best": best, "fitness": best_f}

				# Propose move
				proposal = np.copy(current)
				# Cartesian and rotational steps interleaved
				step_mask = np.ones_like(proposal)
				step_mask[::6] *= cart_step
				step_mask[1::6] *= cart_step
				step_mask[2::6] *= cart_step
				step_mask[3::6] *= rot_step
				step_mask[4::6] *= rot_step
				step_mask[5::6] *= rot_step
				proposal += np.random.uniform(-1, 1, size=proposal.shape) * step_mask

				# Clamp to bounds
				proposal = np.minimum(np.maximum(proposal, lower_bounds.flatten()), upper_bounds.flatten())

				# Limit correction
				proposal = self._limit_correction(proposal[np.newaxis, :], coefficient=(3, 2))[0]

				# Evaluate
				prop_f = self._find_snr(proposal)

				# Metropolis criterion
				accept = (prop_f >= current_f) or (np.random.rand() < np.exp((prop_f - current_f) / max(temp, 1e-9)))
				if accept:
					current = proposal
					current_f = prop_f
					if current_f > best_f:
						best = np.copy(current)
						best_f = current_f
						self.bestFitnessChanged.emit(float(best_f))
						self.bestSolutionChanged.emit(np.copy(best))

				history.append((float(best_f), float(temp)))
				self.progressChanged.emit(int((it + 1) * 100 / n_iterations))

			temp *= cooling_rate

		return {"method": "anneal", "best_solution": best, "best_fitness": float(best_f), "history": history}

	def _run_multi_anneal(self) -> Dict[str, Any]:
		initial_solution = self.data_ctx.get("initial_solution")
		bounds = self.data_ctx.get("bounds")
		if initial_solution is None or bounds is None:
			raise RuntimeError("initial_solution or bounds missing for multianneal")

		lower_bounds, upper_bounds = bounds
		restarts = int(self.params.get("multi_anneal_restarts", 5))
		n_iterations = int(self.params.get("anneal_iterations", 40))
		temp0 = float(self.params.get("anneal_itemp", 100.0))
		min_temp = float(self.params.get("anneal_ftemp", 1e-3))
		cooling_rate = float(self.params.get("anneal_cooling_rate", 0.85))
		cart_step0 = float(self.params.get("anneal_cart_step", 15.0))
		rot_step0 = float(self.params.get("anneal_rot_step", np.pi/3))

		global_best = None
		global_best_f = -np.inf
		all_histories: list = []
		last_best = None
		last_best_f = -np.inf

		for r in range(max(1, restarts)):
			# Reset per-restart state
			current = np.copy(initial_solution).flatten()
			current_f = self._find_snr(current)
			best = np.copy(current)
			best_f = current_f
			temp = temp0
			cart_step = cart_step0
			rot_step = rot_step0
			restart_hist = []

			while temp > min_temp:
				for it in range(n_iterations):
					if self._check_cancel():
						return {"canceled": True, "restart": r, "best": global_best if global_best is not None else best, "fitness": float(max(global_best_f, best_f))}

					proposal = np.copy(current)
					step_mask = np.ones_like(proposal)
					step_mask[::6] *= cart_step
					step_mask[1::6] *= cart_step
					step_mask[2::6] *= cart_step
					step_mask[3::6] *= rot_step
					step_mask[4::6] *= rot_step
					step_mask[5::6] *= rot_step
					proposal += np.random.uniform(-1, 1, size=proposal.shape) * step_mask

					proposal = np.minimum(np.maximum(proposal, lower_bounds.flatten()), upper_bounds.flatten())
					proposal = self._limit_correction(proposal[np.newaxis, :], coefficient=(3, 2))[0]
					prop_f = self._find_snr(proposal)

					accept = (prop_f >= current_f) or (np.random.rand() < np.exp((prop_f - current_f) / max(temp, 1e-9)))
					if accept:
						current = proposal
						current_f = prop_f
						if current_f > best_f:
							best = np.copy(current)
							best_f = current_f
							self.bestFitnessChanged.emit(float(best_f))
							self.bestSolutionChanged.emit(np.copy(best))

					restart_hist.append((float(best_f), float(temp)))
					# Coarse overall progress by restart
					self.progressChanged.emit(int(((r + (it + 1) / n_iterations) * 100) / max(1, restarts)))

				temp *= cooling_rate

			# Update global best
			last_best = np.copy(best)
			last_best_f = float(best_f)
			if best_f > global_best_f:
				global_best_f = float(best_f)
				global_best = np.copy(best)
			all_histories.append(restart_hist)

		return {"method": "multianneal", "best_solution": global_best if global_best is not None else last_best, "best_fitness": float(global_best_f if global_best is not None else last_best_f), "histories": all_histories, "restarts": restarts}

	def _run_gradient_descent(self) -> Dict[str, Any]:
		current = self.data_ctx.get("initial_solution")
		bounds = self.data_ctx.get("bounds")
		if current is None or bounds is None:
			raise RuntimeError("initial_solution or bounds missing for gradient")
		lower_bounds, upper_bounds = bounds

		n_iterations = int(self.params.get("gradient_iterations", 200))
		cart_step = float(self.params.get("gradient_cart_step", 10.0))
		rot_step = float(self.params.get("gradient_rot_step", np.pi/3))
		decay = float(self.params.get("gradient_decay", 0.99))
		simultaneous = int(self.params.get("gradient_simultaneous", -1))

		current = np.copy(current).flatten()
		current_f = self._find_snr(current)
		best = np.copy(current)
		best_f = current_f

		eps = 1e-4
		for it in range(n_iterations):
			if self._check_cancel():
				return {"canceled": True, "iteration": it, "best": best, "fitness": best_f}

			# Build step vector per axis group
			step = np.zeros_like(current)
			axes = list(range(current.shape[0]))
			if simultaneous == -1:
				chosen = axes
			else:
				chosen = axes[:max(1, simultaneous)]

			for ax in chosen:
				# Finite difference gradient
				plus = np.copy(current)
				minus = np.copy(current)
				base_step = cart_step if (ax % 6) < 3 else rot_step
				plus[ax] += eps
				minus[ax] -= eps
				f_plus = self._find_snr(plus)
				f_minus = self._find_snr(minus)
				g = (f_plus - f_minus) / (2 * eps)
				step[ax] = base_step * g

			# Move against gradient
			proposal = current + step
			# Clamp
			proposal = np.minimum(np.maximum(proposal, lower_bounds.flatten()), upper_bounds.flatten())
			# Limit correction
			proposal = self._limit_correction(proposal[np.newaxis, :], coefficient=(3, 2))[0]
			prop_f = self._find_snr(proposal)

			if prop_f > current_f:
				current = proposal
				current_f = prop_f
				if current_f > best_f:
					best = np.copy(current)
					best_f = current_f
					self.bestFitnessChanged.emit(float(best_f))
					self.bestSolutionChanged.emit(np.copy(best))

			# Decay steps
			cart_step *= decay
			rot_step *= decay

			self.progressChanged.emit(int((it + 1) * 100 / n_iterations))

		return {"method": "gradient", "best_solution": best, "best_fitness": float(best_f)}

	def _run_branch_bound(self) -> Dict[str, Any]:
		best = self.data_ctx.get("initial_solution")
		bounds = self.data_ctx.get("bounds")
		if best is None or bounds is None:
			raise RuntimeError("initial_solution or bounds missing for branch/bound")
		lower_bounds, upper_bounds = bounds

		branch_iterations = int(self.params.get("branch_iterations", 50))
		branch_instances = int(self.params.get("branch_instances", 24))
		branch_top = int(self.params.get("branch_top", 6))
		branch_angle_step = float(self.params.get("branch_angle_step", np.pi/4))
		branch_cart_step = float(self.params.get("branch_cart_step", 8.0))
		branch_threshold = float(self.params.get("branch_threshold", 0.1))
		branch_decay = float(self.params.get("branch_decay", 0.95))

		best = np.copy(best).flatten()
		best_f = self._find_snr(best)
		history = []

		for it in range(branch_iterations):
			if self._check_cancel():
				return {"canceled": True, "iteration": it, "best": best, "fitness": best_f}

			# Generate instances around current best
			instances = []
			for _ in range(branch_instances):
				proposal = np.copy(best)
				step_mask = np.ones_like(proposal)
				step_mask[::6] *= branch_cart_step
				step_mask[1::6] *= branch_cart_step
				step_mask[2::6] *= branch_cart_step
				step_mask[3::6] *= branch_angle_step
				step_mask[4::6] *= branch_angle_step
				step_mask[5::6] *= branch_angle_step
				proposal += np.random.uniform(-1, 1, size=proposal.shape) * step_mask
				proposal = np.minimum(np.maximum(proposal, lower_bounds.flatten()), upper_bounds.flatten())
				proposal = self._limit_correction(proposal[np.newaxis, :], coefficient=(3, 2))[0]
				instances.append(proposal)

			fitnesses = np.array([self._find_snr(inst) for inst in instances])
			idx_sorted = np.argsort(-fitnesses)
			top_idxs = idx_sorted[:branch_top]
			best_candidate_f = float(fitnesses[top_idxs[0]])
			best_candidate = np.copy(instances[top_idxs[0]])

			improve_ratio = (best_candidate_f - best_f) / (abs(best_f) + 1e-9)
			history.append((float(best_candidate_f), float(improve_ratio)))
			if improve_ratio > branch_threshold:
				best_f = best_candidate_f
				best = best_candidate
				self.bestFitnessChanged.emit(float(best_f))
				self.bestSolutionChanged.emit(np.copy(best))

			# Decay steps
			branch_cart_step *= branch_decay
			branch_angle_step *= branch_decay

			self.progressChanged.emit(int((it + 1) * 100 / branch_iterations))

		return {"method": "mSGD", "best_solution": best, "best_fitness": float(best_f), "history": history}

	def _run_brute_force(self) -> Dict[str, Any]:
		population = self.data_ctx.get("initial_population")
		bounds = self.data_ctx.get("bounds")
		if population is None or bounds is None:
			raise RuntimeError("initial_population or bounds missing for brute force")

		brute_limit = int(self.params.get("brute_limit", 10_000))
		brute_batch = int(self.params.get("brute_batch", 10))

		evaluated = 0
		best_solution = None
		best_fitness = -np.inf

		for i in range(0, min(brute_limit, population.shape[0]), brute_batch):
			if self._check_cancel():
				return {"canceled": True, "evaluated": evaluated, "best": best_solution, "fitness": best_fitness}

			batch = population[i:i+brute_batch]
			batch = self._limit_correction(batch, coefficient=(3, 2))
			fitnesses = np.array([self._find_snr(chrom) for chrom in batch])

			idx_best = int(np.argmax(fitnesses))
			if fitnesses[idx_best] > best_fitness:
				best_fitness = float(fitnesses[idx_best])
				best_solution = np.copy(batch[idx_best])
				self.bestFitnessChanged.emit(best_fitness)
				self.bestSolutionChanged.emit(np.copy(best_solution))

			evaluated += batch.shape[0]
			self.progressChanged.emit(int(min(100, (evaluated * 100) / max(1, brute_limit))))

		return {"method": "brute", "best_solution": best_solution, "best_fitness": best_fitness, "evaluated": evaluated}


class OptimizerController(QObject):
	"""
	High-level controller for launching optimization runs from the GUI.
	Connect its signals to progress bars, labels, and result views in MDPO_GUI.
	"""

	def __init__(self):
		super().__init__()
		self.worker: Optional[OptimizationWorker] = None
		self.cancel_flag: Dict[str, bool] = {"cancel": False}

	def start(self, method: str, data_ctx: Dict[str, Any], params: Dict[str, Any],
			  on_progress=None, on_status=None, on_best=None, on_finished=None):
		if self.worker and isinstance(self.worker, QThread):
			try:
				if self.worker.isRunning():
					raise RuntimeError("An optimization is already running.")
			except Exception:
				pass

		self.cancel_flag["cancel"] = False
		self.worker = OptimizationWorker(method, data_ctx, params, self.cancel_flag)

		# Wire signals
		if hasattr(self.worker, "progressChanged") and on_progress:
			self.worker.progressChanged.connect(on_progress)
		if hasattr(self.worker, "statusChanged") and on_status:
			self.worker.statusChanged.connect(on_status)
		if hasattr(self.worker, "bestFitnessChanged") and on_best:
			self.worker.bestFitnessChanged.connect(lambda f: on_best("fitness", f))
		if hasattr(self.worker, "bestSolutionChanged") and on_best:
			self.worker.bestSolutionChanged.connect(lambda s: on_best("solution", s))
		if hasattr(self.worker, "finishedWithResult") and on_finished:
			self.worker.finishedWithResult.connect(on_finished)

		# Go
		try:
			self.worker.start()
		except AttributeError:
			# If QThread.start is unavailable in the environment, run synchronously
			self.worker.run()

	def cancel(self):
		self.cancel_flag["cancel"] = True


def build_bounds(n_devices: int,
				 cart_limits: Tuple[np.ndarray, np.ndarray],
				 rot_limits: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Construct flattened lower/upper bounds arrays for optimization variables.
	cart_limits: (lower_xyz[N,3], upper_xyz[N,3]) in MRI units.
	rot_limits: (lower_angles[N,3], upper_angles[N,3]) in radians.
	Returns (lower_bounds[(N*6,)], upper_bounds[(N*6,)]).
	"""
	lower_xyz, upper_xyz = cart_limits
	lower_ang, upper_ang = rot_limits
	lower = np.hstack([lower_xyz, lower_ang]).reshape(n_devices * 6)
	upper = np.hstack([upper_xyz, upper_ang]).reshape(n_devices * 6)
	return lower.astype(float), upper.astype(float)


def build_initial_population(initial_vertices: np.ndarray, k: int) -> np.ndarray:
	"""
	Build an initial population (k solutions) from device origin vertices.
	initial_vertices: (N,3) device origins; angles initialized to zeros.
	Returns array of shape (k, N*6).
	"""
	N = initial_vertices.shape[0]
	base = []
	for i in range(N):
		base.extend(list(initial_vertices[i]))
		base.extend([0.0, 0.0, 0.0])
	base = np.array(base, dtype=float)
	pop = np.tile(base, (k, 1))
	# small random jitter to diversify
	pop += np.random.normal(0, 1e-2, size=pop.shape)
	return pop


def build_data_context(find_snr_fn: Callable[[np.ndarray], float],
					   limit_correction_fn: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]],
					   initial_solution: Optional[np.ndarray],
					   initial_population: Optional[np.ndarray],
					   bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
	"""
	Pack callable hooks and arrays into a data context dict for the worker.
	Any missing optional items will be ignored by methods that do not require them.
	"""
	ctx: Dict[str, Any] = {"find_snr": find_snr_fn}
	if limit_correction_fn is not None:
		ctx["limit_correction"] = limit_correction_fn
	if initial_solution is not None:
		ctx["initial_solution"] = initial_solution
	if initial_population is not None:
		ctx["initial_population"] = initial_population
	if bounds is not None:
		ctx["bounds"] = bounds
	return ctx


def make_find_snr_adapter(recentered_roi: np.ndarray,
						  N_devices: int,
						  transform_vectorspace_fn: Callable[[int, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
						  trim_data_fn: Callable[[int, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
						  calculate_voltage_fn: Callable[[int, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Callable[[np.ndarray], float]:
	"""
	Build a find_snr function bound to GUI data.
	Expects provided functions to implement the corresponding operations used in the script.
	Returns a callable devpos_flat -> total SNR/IC score.
	"""
	def _find_snr(devpos_flat: np.ndarray) -> float:
		devpos = devpos_flat.reshape((N_devices, 6))
		all_dev_volt = np.empty((recentered_roi.shape[0], N_devices))
		all_dev_volt[:] = np.nan
		for idx in range(N_devices):
			dippos, dipvec = transform_vectorspace_fn(idx, recentered_roi, recentered_roi, devpos[idx])
			# If leadfield is needed, try to fetch from a global/context variable attached to the function
			leadfield = getattr(trim_data_fn, "_fields", None)
			if isinstance(leadfield, (list, tuple)) and len(leadfield) > idx:
				lf = leadfield[idx]
			else:
				lf = None
			try:
				dippos, dipvec = trim_data_fn(idx, lf, dippos, dipvec)
			except Exception:
				# Fallback: skip trimming if signature/leadfield are unavailable
				pass
			opt_volt, _, _ = calculate_voltage_fn(idx, dippos, dipvec)
			all_dev_volt[:, idx] = opt_volt
		voltage = np.nanmax(np.nan_to_num(all_dev_volt, nan=0.0), axis=1)
		total = float(np.nansum(voltage))
		return total
	return _find_snr


def make_limit_correction_adapter(angle_limits_rad: Optional[np.ndarray],
								  do_depth: bool,
								  depth_limits_per_dev: Optional[list],
								  recentered_vertices: np.ndarray,
								  proximity_enabled: bool,
								  check_proximity_fn: Optional[Callable[[np.ndarray, int, int], Tuple[float, np.ndarray]]],
								  coefficient_default: Tuple[int, int] = (3, 2)) -> Callable[[np.ndarray, Tuple[int, int]], np.ndarray]:
	"""Thin wrapper that delegates to mdpo_project.make_limit_correction_adapter to avoid duplication."""
	try:
		from mdpo_project import make_limit_correction_adapter as _mk
	except Exception:
		# Fallback to local import path in case of environment differences
		from GUI.mdpo_project import make_limit_correction_adapter as _mk  # type: ignore
	return _mk(angle_limits_rad, do_depth, depth_limits_per_dev, recentered_vertices, proximity_enabled, check_proximity_fn, coefficient_default)

"""Unified MDPO GUI

Tabs:
1. Project Setup: Select brain/ROI .mat files, leadfields, device parameters, save/load project.
2. Optimization: Configure + run genetic optimization (initial integration), load existing results.
3. Assessment: Visualize best solution (device list) and fitness history plot with save option.

This initial scaffold focuses on consolidating workflows. Advanced 3D visualization from
`6_MDPO_visualize.py` and extended assessment notebook logic can be integrated in later passes.
"""

try:
	import matplotlib
	# Prefer interactive Tk backend for embedded viewers; fallback to Agg if unavailable
	try:
		matplotlib.use('TkAgg')
	except Exception:
		matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	MATPLOTLIB_AVAILABLE = True
except ImportError:
	MATPLOTLIB_AVAILABLE = False
	plt = None  # type: ignore
    
# --------------------------------------------------------------
# Device -> Source-space transform helper
# --------------------------------------------------------------
def _transform_device_to_source(devpose: np.ndarray, local_points: np.ndarray) -> np.ndarray:
	"""
	Forward transform: rotate local device points by R(alpha,beta,gamma), then translate by (x,y,z).
	devpose: (6,) [x,y,z,alpha,beta,gamma] already in brain-centered (recentered) coordinates.
	local_points: (M,3) points in device-local frame.
	Returns points in source (brain/ROI) frame.
	"""
	try:
		# Prefer mdpo_project's rotation to stay consistent
		from mdpo_project import get_rotmat
		R = get_rotmat(float(devpose[3]), float(devpose[4]), float(devpose[5]))
	except Exception:
		# Fallback inline rotation
		a, b, g = float(devpose[3]), float(devpose[4]), float(devpose[5])
		yaw = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
		pitch = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
		roll = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
		R = yaw @ pitch @ roll
	pts = (R @ local_points.T).T
	return pts + np.asarray(devpose[:3], dtype=float)

# --------------------------------------------------------------
# Tooltip helper for Tk/ttk widgets
# --------------------------------------------------------------
class _Tooltip:
	def __init__(self, widget, text: str):
		self.widget = widget
		self.text = text
		self.tip = None
		widget.bind('<Enter>', self._show)
		widget.bind('<Leave>', self._hide)

	def _show(self, _event=None):
		if self.tip is not None:
			return
		x = self.widget.winfo_rootx() + 20
		y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
		self.tip = tk.Toplevel(self.widget)
		self.tip.wm_overrideredirect(True)
		self.tip.wm_geometry(f"+{x}+{y}")
		lbl = ttk.Label(self.tip, text=self.text, relief='solid', borderwidth=1, background='#FFFFE0')
		lbl.pack(ipadx=6, ipady=4)

	def _hide(self, _event=None):
		if self.tip is not None:
			try:
				self.tip.destroy()
			except Exception:
				pass
			self.tip = None

def add_tooltip(widget, text: str):
	try:
		_Tooltip(widget, text)
	except Exception:
		pass

# Resolve repository-relative imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # up from GUI/
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
MODULES_DIR = os.path.join(SCRIPTS_DIR, 'modules')
if MODULES_DIR not in sys.path:
	sys.path.insert(0, MODULES_DIR)
try:
	from mdpo_project import (
		Project,
		export_best_solution_csv,
		OptimizationResult,
		SEPIOResult,
		parse_mesh_from_mat,
		normalize_faces,
		vertex_normals,
		inflate_surface,
		outward_angles,
		angle_bins,
		get_palette,
		legend_labels_from_bins,
		map_angles_to_colors,
		# Ensure visualization uses same TF/voltage as optimization
		transform_vectorspace,
		trim_data,
		calculate_voltage,
		make_limit_correction_adapter,
		compute_full_voltage_arrays,
	)
except Exception:
	# Fallback stub to satisfy static analysis; runtime will warn user.
	from dataclasses import dataclass, field
	@dataclass
	class OptimizationResult:  # type: ignore
		method: str = "unknown"
		best_solution: np.ndarray = field(default_factory=lambda: np.zeros(0))
		best_fitness: float = 0.0
		fitness_history: list = field(default_factory=list)
		meta: dict = field(default_factory=dict)

	@dataclass
	class Project:  # type: ignore
		name: str = "Unavailable"
		data_folder: str = ""
		brain_file: str = ""
		roi_files: list = field(default_factory=list)
		roi_labels: list = field(default_factory=list)
		leadfield_files: list = field(default_factory=list)
		device_names: list = field(default_factory=list)
		device_counts: list = field(default_factory=list)
		measure: str = "ic"
		method: str = "mSGD"
		montage: bool = False
		cl_offset: float = 3.0
		do_depth: bool = True
		do_proximity: bool = True
		depth_limits: list = field(default_factory=list)
		proximity_limits: list = field(default_factory=list)
		results: Optional[OptimizationResult] = None
		num_generations: int = 0
		sol_per_pop: int = 0
		num_parents_mating: int = 0
		process_count: int = 0
		ga_mutation_prob: float = 0.1
		angle_limit_rad: list = field(default_factory=list)
		noise: list = field(default_factory=list)
		bandwidth: list = field(default_factory=list)
		scale: list = field(default_factory=list)
		assessment_artifacts: list = field(default_factory=list)  # will hold dicts
		# Physical dimensions per device type (length & radius in mm)
		device_lengths: list = field(default_factory=list)
		device_radii: list = field(default_factory=list)
		anneal_iterations: int = 0
		anneal_itemp: float = 0.0
		anneal_ftemp: float = 0.0
		anneal_cooling_rate: float = 0.0
		anneal_cart_step: float = 0.0
		anneal_rot_step: float = 0.0
		# Multi-anneal
		multi_anneal_restarts: int = 0
		# Gradient stub fields
		gradient_iterations: int = 0
		gradient_cart_step: float = 0.0
		gradient_rot_step: float = 0.0
		gradient_decay: float = 0.0
		gradient_simultaneous: int = -1
		# Branch-bound stub fields
		branch_iterations: int = 0
		branch_instances: int = 0
		branch_top: int = 0
		branch_angle_step: float = 0.0
		branch_cart_step: float = 0.0
		branch_threshold: float = 0.0
		branch_decay: float = 0.0
		# Brute stub fields
		brute_limit: int = 0
		brute_batch: int = 0
		roi_recentered: Optional[np.ndarray] = None
		def save(self, path_out: str): raise RuntimeError("mdpo_project unavailable")
		@staticmethod
		def load(path_in: str): raise RuntimeError("mdpo_project unavailable")
		def run_genetic_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def run_anneal_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def run_gradient_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def run_branch_bound_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def run_brute_force_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def run_optimization(self, progress_callback=None): raise RuntimeError("mdpo_project unavailable")
		def import_result_pickle(self, path: str): raise RuntimeError("mdpo_project unavailable")
		def add_assessment_artifact(self, artifact_type: str, path: str): raise RuntimeError("mdpo_project unavailable")
		def export_epoch_history_csv(self, path_out: str): raise RuntimeError("mdpo_project unavailable")
	def export_best_solution_csv(project, path_out): raise RuntimeError("mdpo_project unavailable")
	def parse_mesh_from_mat(mat_path: str, key_hint: Optional[str] = None): raise RuntimeError("mdpo_project unavailable")
	def normalize_faces(faces, n_vertices): raise RuntimeError("mdpo_project unavailable")
	def vertex_normals(vertices, faces): raise RuntimeError("mdpo_project unavailable")
	def inflate_surface(vertices, normals, dist): raise RuntimeError("mdpo_project unavailable")
	def outward_angles(vertices, normals): raise RuntimeError("mdpo_project unavailable")
	def angle_bins(): return [30,60,90,120,150,180]
	def get_palette(name='Muted'):
		return ['#8C8C8C','#C14343','#C27A44','#C2AD44','#87B640','#44C29C','#446AC2'] if (name or '').lower()=='muted' else ['#8C8C8C','#C23939','#B6C238','#C26138','#38C256','#38C0C2','#3858C2']
	def legend_labels_from_bins(bins):
		labels=[]; prev=0
		for b in bins:
			labels.append((f"≤ {int(b)}°") if prev==0 else (f"{int(prev)}–{int(b)}°")); prev=b
		return labels
	def map_angles_to_colors(vals, bins, palette):
		out=[]
		for a in vals:
			c=palette[0]
			try:
				if not np.isnan(a):
					for i,th in enumerate(bins, start=1):
						if a<=th: c=palette[min(i,len(palette)-1)]; break
					else:
						c=palette[-1]
			except Exception:
				pass
			out.append(c)
		return out
	def compute_full_voltage_arrays(project, solution):
		raise RuntimeError("mdpo_project unavailable")


class MDPOGUI(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("MDPO Unified Interface")
		self.geometry("1100x1100")

		# Windows-classic aesthetic: safe ttk style setup
		try:
			style = ttk.Style()
			themes = style.theme_names()
			# Prefer classic/winnative themes for a 9x look
			if 'classic' in themes:
				style.theme_use('classic')
			elif 'winnative' in themes:
				style.theme_use('winnative')
			# Windows 95 palette and element tweaks
			bg = '#C0C0C0'   # SystemButtonFace
			fg = 'black'
			btn_bg = '#D9D9D9'
			light = '#FFFFFF'
			shadow = '#808080'
			dark = '#404040'
			# Root background
			self.configure(bg=bg)
			# Frames and group boxes
			style.configure('TLabelframe', background=bg, borderwidth=2, relief='groove')
			style.configure('TLabelframe.Label', background=bg, foreground=fg)
			style.configure('TFrame', background=bg)
			# Labels and checkbuttons
			style.configure('TLabel', background=bg, foreground=fg)
			style.configure('TCheckbutton', background=bg, foreground=fg)
			# Buttons: raised default, sunken pressed
			style.configure('TButton', background=btn_bg, foreground=fg, padding=(6,2))
			style.map('TButton', relief=[('pressed','sunken'), ('active','raised')],
				background=[('active', '#E5E5E5')])
			# Entries: white fieldbackground with slight padding
			try:
				style.configure('TEntry', fieldbackground='#FFFFFF', background='#FFFFFF', padding=(2,1))
			except Exception:
				pass
			# Notebook and tabs: raised tabs, selected looks sunken
			style.configure('TNotebook', background=bg, borderwidth=2)
			style.configure('TNotebook.Tab', background=btn_bg, padding=(10,4))
			style.map('TNotebook.Tab', relief=[('selected','sunken'), ('!selected','raised')])
			# Treeview: white background, flat headers
			style.configure('Treeview', background='#FFFFFF', fieldbackground='#FFFFFF', foreground=fg)
			style.configure('Treeview.Heading', background=btn_bg, relief='raised')
			# Progressbar: muted trough and teal-ish bar
			style.configure('Horizontal.TProgressbar', troughcolor='#DFDFDF')
			# Classic small fonts
			default_font = ('MS Sans Serif', 9)
			self.option_add('*Font', default_font)
			self.option_add('*Entry*Font', default_font)
			self.option_add('*Text*Font', default_font)
			self.option_add('*Button*Font', default_font)
		except Exception:
			# If style setup fails, continue with defaults
			pass
		# Menubar and status bar (Windows 95 style)
		self._build_menubar()
		self.status_var = tk.StringVar(value='Ready')
		self.status_bar = tk.Label(self, textvariable=self.status_var, anchor='w', bg='#C0C0C0', fg='black', bd=2, relief='sunken')
		self.status_bar.pack(side='bottom', fill='x')
		self.project: Optional[Project] = None
		# Shared variables used across tabs
		self.var_measure = tk.StringVar(value="ic")
		self.var_gens = tk.StringVar(value="30")
		self.var_pop = tk.StringVar(value="44")
		self.var_parents = tk.StringVar(value="15")
		self.var_proc = tk.StringVar(value="4")
		self.var_mut_prob = tk.StringVar(value="0.1")
		# Gradient params
		self.var_gd_iter = tk.StringVar(value="200")
		self.var_gd_cart = tk.StringVar(value="10")
		self.var_gd_rot = tk.StringVar(value=f"{np.pi/3:.3f}")
		self.var_gd_decay = tk.StringVar(value="0.99")
		self.var_gd_simult = tk.StringVar(value="-1")
		# Branch-bound params
		self.var_bb_iter = tk.StringVar(value="36")
		self.var_bb_instances = tk.StringVar(value="24")
		self.var_bb_top = tk.StringVar(value="6")
		self.var_bb_angle = tk.StringVar(value=f"{np.pi/4:.3f}")
		self.var_bb_cart = tk.StringVar(value="8")
		self.var_bb_thresh = tk.StringVar(value="0.1")
		self.var_bb_decay = tk.StringVar(value="0.95")
		# Brute force params
		self.var_bf_limit = tk.StringVar(value="1000")
		self.var_bf_batch = tk.StringVar(value="10")
		# Structural Avoidance (GUI-only for now)
		self.var_struct_demo = tk.BooleanVar(value=False)
		self.var_struct_radius = tk.StringVar(value="5.0")
		# Backend storage for structural avoidance spheres [x,y,z,r]
		self.structural_avoidance: Optional[np.ndarray] = None
		# SEPIO session storage
		self.sepio_voltage: Optional[np.ndarray] = None
		self.sepio_ic: Optional[np.ndarray] = None
		self.sepio_current_result: Optional[OptimizationResult] = None
		self._build_notebook()

	# --------------------------------------------------------------
	# Notebook and Tabs
	# --------------------------------------------------------------
	def _build_notebook(self):
		self.nb = ttk.Notebook(self)
		self.tab_setup = ttk.Frame(self.nb)
		self.tab_opt = ttk.Frame(self.nb)
		self.tab_assess = ttk.Frame(self.nb)
		self.tab_sepio = ttk.Frame(self.nb)
		self.tab_plot = ttk.Frame(self.nb)
		self.nb.add(self.tab_setup, text="1. Project Setup")
		self.nb.add(self.tab_opt, text="2. Optimization")
		self.nb.add(self.tab_assess, text="3. Assessment")
		self.nb.add(self.tab_sepio, text="4. SEPIO")
		self.nb.add(self.tab_plot, text="5. Plotting")
		self.nb.pack(fill='both', expand=True)
		self._build_setup_tab()
		self._build_opt_tab()
		self._build_assess_tab()
		self._build_sepio_tab()
		self._build_plot_tab()
		self._set_status('Ready')

	# --------------------------------------------------------------
	# Plotting Tab (Tab 5)
	# --------------------------------------------------------------
	def _build_plot_tab(self):
		frm = self.tab_plot
		# Remembered selections live on the Project as a list of SEPIO result labels
		self.var_plot_select = tk.StringVar(value='')

		grp = ttk.LabelFrame(frm, text="Select Datasets")
		grp.pack(fill='x', padx=8, pady=8)

		row = 0
		ttk.Label(grp, text="SEPIO Result:").grid(row=row, column=0, sticky='w', padx=6, pady=6)
		self.cb_plot_results = ttk.Combobox(grp, textvariable=self.var_plot_select, state='readonly', width=50, values=self._plot_available_result_labels())
		self.cb_plot_results.grid(row=row, column=1, sticky='we', padx=6, pady=6)
		btn_add = ttk.Button(grp, text="Add", command=self._plot_add_dataset)
		btn_add.grid(row=row, column=2, sticky='w', padx=6, pady=6)
		grp.columnconfigure(1, weight=1)

		row += 1
		# Selected list
		lst_frame = ttk.Frame(grp)
		lst_frame.grid(row=row, column=0, columnspan=3, sticky='nsew', padx=6, pady=(0,6))
		grp.rowconfigure(row, weight=1)
		lst_frame.columnconfigure(0, weight=1)
		self.lb_plot_selected = tk.Listbox(lst_frame, height=8, selectmode='extended')
		self.lb_plot_selected.grid(row=0, column=0, sticky='nsew')
		sb = ttk.Scrollbar(lst_frame, orient='vertical', command=self.lb_plot_selected.yview)
		sb.grid(row=0, column=1, sticky='ns')
		self.lb_plot_selected.configure(yscrollcommand=sb.set)

		row += 1
		btns = ttk.Frame(grp)
		btns.grid(row=row, column=0, columnspan=3, sticky='w', padx=6, pady=6)
		btn_remove = ttk.Button(btns, text="Remove", command=self._plot_remove_selected)
		btn_remove.pack(side='left', padx=(0,8))
		btn_save = ttk.Button(btns, text="Save Project", command=self._save_project)
		btn_save.pack(side='left')

		# Initial population of listbox from project (if available)
		self._plot_refresh_listbox()

		# Plotting actions
		plot_grp = ttk.LabelFrame(frm, text="Plot SEPIO Results")
		plot_grp.pack(fill='x', padx=8, pady=8)
		btn_line = ttk.Button(plot_grp, text="Accuracy Line Plot", command=self._plot_accuracy_lines)
		btn_line.grid(row=0, column=0, padx=6, pady=6, sticky='w')
		# New heatmap button beside the existing line plot
		btn_heat = ttk.Button(plot_grp, text="Accuracy Heatmap (Sensors X Devices)", command=self._plot_accuracy_heatmap_sensors_devices)
		btn_heat.grid(row=0, column=1, padx=6, pady=6, sticky='w')

	def _plot_available_result_labels(self):
		if self.project is None or not getattr(self.project, 'sepio_results', None):
			return []
		try:
			return [sr.label for sr in self.project.sepio_results]
		except Exception:
			return []

	def _plot_refresh_dropdown(self):
		if hasattr(self, 'cb_plot_results'):
			self.cb_plot_results['values'] = self._plot_available_result_labels()
			# Clear selection if it no longer exists
			if self.var_plot_select.get() and self.var_plot_select.get() not in self.cb_plot_results['values']:
				self.var_plot_select.set('')

	def _plot_refresh_listbox(self):
		if not hasattr(self, 'lb_plot_selected'):
			return
		self.lb_plot_selected.delete(0, tk.END)
		labels = []
		if self.project is not None and getattr(self.project, 'plotting_datasets', None):
			labels = list(self.project.plotting_datasets)
		for lbl in labels:
			self.lb_plot_selected.insert(tk.END, lbl)
		self._plot_refresh_dropdown()

	def _plot_add_dataset(self):
		label = (self.var_plot_select.get() or '').strip()
		if not label:
			messagebox.showwarning("Select Dataset", "Please select a SEPIO result to add.")
			return
		if self.project is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		if not hasattr(self.project, 'plotting_datasets') or self.project.plotting_datasets is None:
			self.project.plotting_datasets = []
		if label in self.project.plotting_datasets:
			messagebox.showinfo("Already Added", f"'{label}' is already in the selection list.")
			return
		self.project.plotting_datasets.append(label)
		self._plot_refresh_listbox()
		self._set_status(f"Added dataset '{label}' to plotting selection")

	def _plot_remove_selected(self):
		if self.project is None or not getattr(self.project, 'plotting_datasets', None):
			return
		sel = list(self.lb_plot_selected.curselection())
		if not sel:
			return
		# Remove in reverse index order
		for idx in reversed(sel):
			try:
				label = self.lb_plot_selected.get(idx)
				dself = self.project.plotting_datasets
				if label in dself:
					dself.remove(label)
			except Exception:
				pass
		self._plot_refresh_listbox()
		self._set_status("Removed selected plotting datasets")

	def _plot_accuracy_lines(self):
		# Preconditions
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		labels = list(getattr(proj, 'plotting_datasets', []) or [])
		if not labels:
			messagebox.showinfo("No Datasets", "Use 'Add' to select SEPIO results for plotting.")
			return
		# Resolve SEPIO results by label
		results_map: Dict[str, Any] = {}
		for ent in getattr(proj, 'sepio_results', []) or []:
			lab = getattr(ent, 'label', '')
			if lab in labels:
				results_map[lab] = ent
		# Filter missing
		labels = [lab for lab in labels if lab in results_map]
		if not labels:
			messagebox.showinfo("No Data", "Selected labels not found among saved SEPIO results.")
			return
		# Sort by max sensors available (descending)
		def _max_sensors(ent):
			try:
				sr = np.asarray(getattr(ent, 'sensor_range', np.array([])))
				return int(np.max(sr)) if sr.size else 0
			except Exception:
				return 0
		labels.sort(key=lambda lab: _max_sensors(results_map[lab]), reverse=True)
		# Build line plot
		if not self._matplotlib_required():
			return
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(figsize=(8, 5))
		colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(labels))))
		for i, lab in enumerate(labels):
			ent = results_map[lab]
			sensors = np.asarray(getattr(ent, 'sensor_range', np.array([])))
			accs = np.asarray(getattr(ent, 'accs_avg', np.array([])))
			if sensors.size == 0 or accs.size == 0:
				continue
			# Ensure ascending sensor order
			idx = np.argsort(sensors)
			x = sensors[idx]
			y = accs[idx]
			ax.plot(x, y, lw=2, color=colors[i], label=lab, marker='o', markersize=4, markerfacecolor='white', markeredgecolor=colors[i])
		ax.set_xlabel('Sensors')
		ax.set_ylabel('Accuracy')
		ax.set_title('SEPIO Accuracy vs Sensors')
		# Small-text legend
		leg = ax.legend(loc='best', fontsize=8)
		fig.tight_layout()
		# Show popup with PNG preview and PDF save option
		self._show_matplotlib_window_with_save(fig, title="Accuracy Line Plot", save_label="Save PDF to Project", prefix="sepio_accuracy")

	def _plot_accuracy_heatmap_sensors_devices(self):
		# Preconditions
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		labels = list(getattr(proj, 'plotting_datasets', []) or [])
		if not labels:
			messagebox.showinfo("No Datasets", "Use 'Add' to select SEPIO results for plotting.")
			return
		# Resolve SEPIO results by label
		results_map: Dict[str, Any] = {}
		for ent in getattr(proj, 'sepio_results', []) or []:
			lab = getattr(ent, 'label', '')
			if lab in labels:
				results_map[lab] = ent
		# Filter missing
		labels = [lab for lab in labels if lab in results_map]
		if not labels:
			messagebox.showinfo("No Data", "Selected labels not found among saved SEPIO results.")
			return

		# Build union of sensor counts (x-axis) and device counts (y-axis)
		try:
			device_counts = sorted({int(getattr(results_map[lab], 'n_devices', 0)) for lab in labels})
			sensor_union = sorted({int(s) for lab in labels for s in np.asarray(getattr(results_map[lab], 'sensor_range', np.array([])))})
			if not device_counts or not sensor_union:
				messagebox.showinfo("No Data", "Selected SEPIO results lack sensor/device metadata.")
				return
		except Exception:
			messagebox.showerror("Data Error", "Failed to parse sensor/device ranges from selected results.")
			return

		# Initialize matrix with NaNs (rows: devices, cols: sensors)
		mat = np.full((len(device_counts), len(sensor_union)), np.nan, dtype=float)
		# Fill matrix with available accuracies
		for lab in labels:
			ent = results_map[lab]
			n_dev = int(getattr(ent, 'n_devices', 0))
			sensors = np.asarray(getattr(ent, 'sensor_range', np.array([])))
			accs = np.asarray(getattr(ent, 'accs_avg', np.array([])))
			if sensors.size == 0 or accs.size == 0:
				continue
			# Ensure sorted by sensors
			idx = np.argsort(sensors)
			sensors_sorted = sensors[idx].astype(int)
			accs_sorted = accs[idx]
			# Row index for this device count
			try:
				row_i = device_counts.index(n_dev)
			except ValueError:
				continue
			# Place values in appropriate columns
			for s_val, a_val in zip(sensors_sorted, accs_sorted):
				try:
					col_j = sensor_union.index(int(s_val))
					mat[row_i, col_j] = float(a_val)
				except ValueError:
					pass

		# Plot blended heatmap using tricontourf
		if not self._matplotlib_required():
			return
		import matplotlib.pyplot as plt
		import matplotlib.tri as mtri
		fig, ax = plt.subplots(figsize=(8, 5))
		cmap = plt.cm.viridis
		# Collect valid points (x: sensors, y: devices, z: accuracy)
		x_pts = []
		y_pts = []
		z_vals = []
		for i, d in enumerate(device_counts):
			for j, s in enumerate(sensor_union):
				val = mat[i, j]
				if not np.isnan(val):
					x_pts.append(float(s))
					y_pts.append(float(d))
					z_vals.append(float(val))
		if len(z_vals) < 3:
			messagebox.showinfo("Insufficient Data", "Need at least 3 points to build a contour plot.")
			return
		# Triangulate and draw tricontourf
		triang = mtri.Triangulation(x_pts, y_pts)
		levels = 15
		cntr = ax.tricontourf(triang, z_vals, levels=levels, cmap=cmap)
		# Overlay points for clarity
		ax.plot(x_pts, y_pts, 'o', ms=3, color='white', alpha=0.7)
		# Axes and colorbar
		ax.set_xlabel('Sensors')
		ax.set_ylabel('Devices')
		ax.set_title('SEPIO Accuracy (tricontourf) — Sensors × Devices')
		cbar = fig.colorbar(cntr, ax=ax)
		cbar.set_label('Accuracy')
		fig.tight_layout()
		self._show_matplotlib_window_with_save(fig, title="Accuracy - Sensors X Devices", save_label="Save PDF to Project", prefix="sepio_accuracy_heatmap")

	def _show_matplotlib_window_with_save(self, fig, title: str, save_label: str, prefix: str):
		"""Display a Matplotlib figure in a popup and provide a Save-to-PDF button.

		- Saves a temporary PNG for preview; closing the fig after.
		- On save, writes a PDF copy into the project folder and records artifact.
		"""
		try:
			png_path = self._plot_to_temp_png(fig, dpi=120)
			win = tk.Toplevel(self)
			win.title(title)
			img = tk.PhotoImage(file=png_path)
			lbl = ttk.Label(win, image=img)
			lbl.image = img
			lbl.pack(padx=8, pady=8)
			def _save_pdf():
				if not self._ensure_project_and_folder():
					return
				try:
					import datetime
					stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
					pdf_path = os.path.join(self.project.data_folder, f"{prefix}_{stamp}.pdf")
					fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
					self._record_artifact('sepio_accuracy_pdf', pdf_path)
					messagebox.showinfo("Saved", f"Saved PDF: {os.path.basename(pdf_path)}")
				except Exception as e:
					messagebox.showerror("Save Error", f"Failed to save PDF: {e}")
			btn = ttk.Button(win, text=save_label, command=_save_pdf)
			btn.pack(pady=6)
			return win
		except Exception as e:
			messagebox.showerror("Plot Error", f"Failed to render plot: {e}")

	# --------------------------------------------------------------
	# Setup Tab
	# --------------------------------------------------------------
	def _build_setup_tab(self):
		frm = self.tab_setup
		# Project name & folder
		top = ttk.LabelFrame(frm, text="Project Metadata")
		top.pack(fill='x', padx=8, pady=6)
		ttk.Label(top, text="Name:").grid(row=0, column=0, sticky='w')
		self.var_name = tk.StringVar(value="Untitled")
		ttk.Entry(top, textvariable=self.var_name, width=30).grid(row=0, column=1, sticky='w')
		ttk.Label(top, text="Data Folder:").grid(row=1, column=0, sticky='w')
		self.var_folder = tk.StringVar(value="")
		ttk.Entry(top, textvariable=self.var_folder, width=60).grid(row=1, column=1, sticky='w')
		ttk.Button(top, text="Browse", command=lambda: self._browse_dir(self.var_folder)).grid(row=1, column=2, padx=4)

		# Brain file
		brain_box = ttk.LabelFrame(frm, text="Brain Surface (.mat)")
		brain_box.pack(fill='x', padx=8, pady=4)
		self.var_brain = tk.StringVar()
		ttk.Entry(brain_box, textvariable=self.var_brain, width=80).grid(row=0, column=0, sticky='w')
		ttk.Button(brain_box, text="Select", command=lambda: self._browse_file(self.var_brain, ext=("MAT files","*.mat"))).grid(row=0, column=1)

		# ROI files list
		roi_box = ttk.LabelFrame(frm, text="ROI Surfaces (.mat)")
		roi_box.pack(fill='x', padx=8, pady=4)
		self.lb_roi = tk.Listbox(roi_box, height=5, selectmode='extended')
		self.lb_roi.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.lb_roi.grid(row=0, column=0, sticky='nsew')
		roi_box.grid_columnconfigure(0, weight=1)
		roi_btns = ttk.Frame(roi_box)
		roi_btns.grid(row=0, column=1, sticky='ns')
		ttk.Button(roi_btns, text="Add", command=self._add_roi).pack(fill='x', pady=2)
		ttk.Button(roi_btns, text="Remove", command=self._remove_roi).pack(fill='x', pady=2)

		# Structural Avoidance (In development)
		struct_box = ttk.LabelFrame(frm, text="Structural Avoidance (In development)")
		struct_box.pack(fill='x', padx=8, pady=4)
		self.lb_struct = tk.Listbox(struct_box, height=5, selectmode='extended')
		self.lb_struct.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.lb_struct.grid(row=0, column=0, sticky='nsew')
		struct_box.grid_columnconfigure(0, weight=1)
		struct_btns = ttk.Frame(struct_box)
		struct_btns.grid(row=0, column=1, sticky='ns')
		ttk.Button(struct_btns, text="Add", command=self._add_struct).pack(fill='x', pady=2)
		ttk.Button(struct_btns, text="Remove", command=self._remove_struct).pack(fill='x', pady=2)
		# Options row under file selection
		struct_opts = ttk.Frame(struct_box)
		struct_opts.grid(row=1, column=0, columnspan=2, sticky='w', pady=4)
		ttk.Checkbutton(struct_opts, text="Demo @ ROI", variable=self.var_struct_demo).pack(side='left', padx=4)
		ttk.Label(struct_opts, text="Radius (mm):").pack(side='left', padx=(12,4))
		ttk.Entry(struct_opts, textvariable=self.var_struct_radius, width=8).pack(side='left')

		# Leadfield files
		lf_box = ttk.LabelFrame(frm, text="Lead Field Files (.npz)")
		lf_box.pack(fill='x', padx=8, pady=4)
		# Treeview with Type column to match Device Configuration order
		self.tv_lf = ttk.Treeview(lf_box, columns=('type','path'), show='headings', height=4)
		self.tv_lf.heading('type', text='Type')
		self.tv_lf.heading('path', text='Path')
		self.tv_lf.column('type', width=80, anchor='w')
		self.tv_lf.column('path', width=600, anchor='w')
		self.tv_lf.grid(row=0, column=0, sticky='nsew')
		lf_box.grid_columnconfigure(0, weight=1)
		lf_btns = ttk.Frame(lf_box)
		lf_btns.grid(row=0, column=1, sticky='ns')
		ttk.Button(lf_btns, text="Add", command=self._add_lf).pack(fill='x', pady=2)
		ttk.Button(lf_btns, text="Remove", command=self._remove_lf).pack(fill='x', pady=2)

		# Unified per-device grid based on Lead Field files
		self.devices_box = ttk.LabelFrame(frm, text="Device Configuration (per Lead Field)")
		self.devices_box.pack(fill='x', padx=8, pady=6)
		self.devices_grid = ttk.Frame(self.devices_box)
		self.devices_grid.pack(fill='x', padx=6, pady=6)
		self.var_do_depth = tk.BooleanVar(value=True)
		self.var_do_prox = tk.BooleanVar(value=True)
		controls = ttk.Frame(self.devices_box)
		controls.pack(fill='x', padx=6, pady=6)
		self.btn_update_devices = ttk.Button(controls, text="Update", command=self._rebuild_device_grid)
		self.btn_update_devices.pack(side='left', padx=4)
		ttk.Checkbutton(controls, text="Depth Enabled", variable=self.var_do_depth).pack(side='left', padx=8)
		ttk.Checkbutton(controls, text="Proximity Enabled", variable=self.var_do_prox).pack(side='left', padx=8)
		self.lbl_devices_status = ttk.Label(controls, text="No leadfields yet")
		self.lbl_devices_status.pack(side='left', padx=12)
		# Per-column entry widgets (lists of entries)
		self.dev_name_entries = []
		self.dev_count_entries = []
		self.dev_noise_entries = []
		self.dev_bw_entries = []
		self.dev_scale_entries = []
		self.dev_angle_entries = []
		self.dev_depth_min_entries = []
		self.dev_depth_max_entries = []
		self.dev_prox_entries = []
		# New physical dimension fields
		self.dev_length_entries = []
		self.dev_radius_entries = []

		# Save / Load project
		actions = ttk.Frame(frm)
		actions.pack(fill='x', padx=8, pady=10)
		ttk.Button(actions, text="Create Project / Clear Results", command=self._create_or_update_project).pack(side='left', padx=4)
		ttk.Button(actions, text="Save Project", command=self._save_project).pack(side='left', padx=4)
		ttk.Button(actions, text="Load Project", command=self._load_project).pack(side='left', padx=4)
		ttk.Button(actions, text="Clear Form", command=self._clear_form).pack(side='left', padx=4)
		self.lbl_status_setup = ttk.Label(actions, text="Ready")
		self.lbl_status_setup.pack(side='left', padx=12)

	# --------------------------------------------------------------
	# Optimization Tab
	# --------------------------------------------------------------
	def _build_opt_tab(self):
		frm = self.tab_opt
		self.lbl_proj_summary = ttk.LabelFrame(frm, text="Current Project Summary")
		self.lbl_proj_summary.pack(fill='x', padx=8, pady=6)
		self.txt_summary = tk.Text(self.lbl_proj_summary, height=12, wrap='word')
		self.txt_summary.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.txt_summary.pack(fill='both', expand=True)
		self.btn_refresh_summary = ttk.Button(frm, text="Refresh Summary", command=self._refresh_summary)
		self.btn_refresh_summary.pack(padx=8, pady=4, anchor='w')

		run_box = ttk.LabelFrame(frm, text="Optimization Controls")
		run_box.pack(fill='x', padx=8, pady=6)
		self.var_run_method = tk.StringVar(value="mSGD")
		ttk.Label(run_box, text="Method:").grid(row=0, column=0, sticky='w')
		cmb_method = ttk.Combobox(run_box, textvariable=self.var_run_method, values=["mSGD","genetic","anneal","multianneal","brute"], width=12, state='readonly')
		cmb_method.grid(row=0, column=1, sticky='w')
		ttk.Label(run_box, text="Measure:").grid(row=0, column=2, sticky='w')
		ttk.Combobox(run_box, textvariable=self.var_measure, values=["ic","snr","voltage"], width=12, state='readonly').grid(row=0, column=3, sticky='w')
		self.btn_run = ttk.Button(run_box, text="Start", command=self._start_optimization)
		self.btn_run.grid(row=0, column=4, padx=6)
		self.lbl_opt_status = ttk.Label(run_box, text="Idle")
		self.lbl_opt_status.grid(row=0, column=5, padx=6)
		# Progress bar for real-time updates
		self.progress = ttk.Progressbar(run_box, orient='horizontal', length=400, mode='determinate')
		self.progress.grid(row=1, column=0, columnspan=6, pady=6, sticky='w')

		# Dynamic Parameter Selection block
		self.param_box = ttk.LabelFrame(frm, text="Parameter Selection")
		self.param_box.pack(fill='x', padx=8, pady=6)
		# Common parameter: Initial Solutions count
		self.var_init_solutions = tk.StringVar(value="100")
		# Anneal parameter variables
		self.var_a_iter = tk.StringVar(value="35")
		self.var_a_itemp = tk.StringVar(value="100")
		self.var_a_ftemp = tk.StringVar(value="0.001")
		self.var_a_cool = tk.StringVar(value="0.6")
		self.var_a_cart = tk.StringVar(value="15")
		self.var_a_rot = tk.StringVar(value=f"{np.pi/3:.3f}")
		# Multi-anneal restarts
		self.var_ma_restarts = tk.StringVar(value="5")
		# Build initial selection for default method
		self._rebuild_param_selection()
		cmb_method.bind('<<ComboboxSelected>>', lambda e: self._rebuild_param_selection())

		# Removed 'Load Existing Results' section from Optimization tab per request

		# Optimization Results management
		del_box = ttk.LabelFrame(frm, text="Remove Optimization Results")
		del_box.pack(fill='x', padx=8, pady=6)
		self.var_del_result = tk.StringVar()
		self.cmb_del_result = ttk.Combobox(del_box, textvariable=self.var_del_result, values=[], width=60, state='readonly')
		self.cmb_del_result.grid(row=0, column=0, sticky='w', padx=4, pady=4)
		btn_del = ttk.Button(del_box, text="Delete", command=self._delete_selected_result)
		btn_del.grid(row=0, column=1, padx=6, pady=4)
		# Add Update List button to reload existing optimization result sets
		btn_update = ttk.Button(del_box, text="Update List", command=self._refresh_delete_list)
		btn_update.grid(row=0, column=2, padx=6, pady=4)
		# Clear All button: removes all saved optimization results (with double confirmation)
		btn_clear_all = ttk.Button(del_box, text="Clear All", command=self._clear_all_results)
		btn_clear_all.grid(row=0, column=3, padx=6, pady=4)
		# Populate list initially
		self._refresh_delete_list()

	# --------------------------------------------------------------
	# Assessment Tab
	# --------------------------------------------------------------
	def _build_assess_tab(self):
		frm = self.tab_assess
		# Results selection dropdown
		sel_box = ttk.LabelFrame(frm, text="Select Optimization Result")
		sel_box.pack(fill='x', padx=8, pady=6)
		self.var_result_sel = tk.StringVar()
		self.cmb_result_sel = ttk.Combobox(sel_box, textvariable=self.var_result_sel, values=[], width=60, state='readonly')
		self.cmb_result_sel.grid(row=0, column=0, sticky='w')
		self.cmb_result_sel.bind('<<ComboboxSelected>>', lambda e: self._on_select_result())
		ttk.Button(sel_box, text="Refresh List", command=self._refresh_result_list).grid(row=0, column=1, padx=6)

		top = ttk.LabelFrame(frm, text="Optimization Result Details")
		top.pack(fill='x', padx=8, pady=6)
		self.txt_result = tk.Text(top, height=10, wrap='word')
		self.txt_result.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.txt_result.pack(fill='both', expand=True)
		ttk.Button(top, text="Refresh", command=self._refresh_result).pack(anchor='w', padx=4, pady=4)

		plot_box = ttk.LabelFrame(frm, text="Assessment Plots")
		plot_box.pack(fill='x', padx=8, pady=6)
		ttk.Button(plot_box, text="Fitness Plot", command=self._show_fitness_plot).grid(row=0, column=0, padx=4, pady=4)
		ttk.Button(plot_box, text="Visualize Best", command=self._visualize_best_open3d).grid(row=0, column=1, padx=4, pady=4)
		# Only ROI: default True
		self.var_only_roi = tk.IntVar(value=1)
		self.chk_only_roi = ttk.Checkbutton(plot_box, text="Only ROI", variable=self.var_only_roi)
		self.chk_only_roi.grid(row=0, column=2, padx=4, pady=4)
		# Show Avoidance: default True
		self.var_show_avoidance = tk.IntVar(value=1)
		self.chk_show_avoidance = ttk.Checkbutton(plot_box, text="Show Avoidance", variable=self.var_show_avoidance)
		self.chk_show_avoidance.grid(row=0, column=3, padx=4, pady=4)
		# Debug toggle
		self.var_debug = tk.IntVar(value=0)
		self.chk_debug = ttk.Checkbutton(plot_box, text="Debug", variable=self.var_debug)
		self.chk_debug.grid(row=0, column=4, padx=4, pady=4)
		# X-Ray toggle: renders brain outer surface semi-transparent and double-sided
		self.var_xray = tk.IntVar(value=0)
		self.chk_xray = ttk.Checkbutton(plot_box, text="X-Ray View", variable=self.var_xray)
		self.chk_xray.grid(row=0, column=5, padx=4, pady=4)
		self.lbl_assess_status = ttk.Label(plot_box, text="Ready")
		self.lbl_assess_status.grid(row=0, column=6, padx=8)
		# (Save OBJ toggle removed)

		# Exports block moved below plotting
		export_box = ttk.LabelFrame(frm, text="Exports")
		export_box.pack(fill='x', padx=8, pady=6)
		ttk.Button(export_box, text="Export Best Solution CSV", command=self._export_best_csv).grid(row=0, column=0, padx=4, pady=4)
		ttk.Button(export_box, text="Export all results by epoch", command=self._export_epoch_csv).grid(row=0, column=1, padx=4, pady=4)

		# Initialize list on build
		self._refresh_result_list()

	# --------------------------------------------------------------
	# SEPIO Tab
	# --------------------------------------------------------------
	def _build_sepio_tab(self):
		frm = self.tab_sepio
		# Selection block
		box_sel = ttk.LabelFrame(frm, text="Select Optimization Result")
		box_sel.pack(fill='x', padx=8, pady=6)
		self.cmb_sepio = ttk.Combobox(box_sel, state='readonly', width=80)
		self.cmb_sepio.grid(row=0, column=0, sticky='w')
		self.cmb_sepio.bind('<<ComboboxSelected>>', lambda _e: self._sepio_on_select())
		ttk.Button(box_sel, text="Refresh", command=self._sepio_refresh_results).grid(row=0, column=1, padx=6)

		# Details block
		box_det = ttk.LabelFrame(frm, text="Result Details")
		box_det.pack(fill='x', padx=8, pady=6)
		self.lbl_sepio_details = ttk.Label(box_det, text="No result selected.")
		self.lbl_sepio_details.pack(anchor='w', padx=6, pady=4)

		# Compute & plots
		box_cmp = ttk.LabelFrame(frm, text="Compute & Plot Voltage")
		box_cmp.pack(fill='x', padx=8, pady=8)
		btn_recompute = ttk.Button(box_cmp, text="Re-compute Voltages", command=self._sepio_recompute_voltages)
		btn_recompute.grid(row=0, column=0, padx=4, pady=4, sticky='w')
		add_tooltip(btn_recompute, "Compute per-source, per-sensor voltage and IC; stored in session.")
		btn_hist_src = ttk.Button(box_cmp, text="Plot Voltage Distribution (sources)", command=self._sepio_plot_voltage_sources)
		btn_hist_src.grid(row=0, column=1, padx=4, pady=4, sticky='w')
		add_tooltip(btn_hist_src, "Histogram of max |V| across sensors for each source.")

		# Summary panel (match size of Optimization Result Details)
		box_sum = ttk.LabelFrame(frm, text="Voltage Summary")
		box_sum.pack(fill='x', padx=8, pady=8)
		self.txt_sepio_summary = tk.Text(box_sum, height=10, wrap='word')
		self.txt_sepio_summary.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.txt_sepio_summary.pack(fill='both', expand=True)
		self.txt_sepio_summary.insert('1.0', 'No voltage computed yet. Press "Re-compute Voltages" to populate this summary.')
		self.txt_sepio_summary.config(state='disabled')

		# SEPIO Monte-Carlo controls
		box_mc = ttk.LabelFrame(frm, text="SEPIO Monte-Carlo")
		box_mc.pack(fill='x', padx=8, pady=8)
		# Parameters
		self.var_mc_cycles = tk.StringVar(value="5")
		self.var_sensor_div = tk.StringVar(value="16")
		self.var_noise_std = tk.StringVar(value="0.01")
		self.var_nbasis = tk.StringVar(value="20")
		self.var_l1 = tk.StringVar(value="0.001")
		self.var_rep_per_max = tk.StringVar(value="1")
		self.var_rep_subdiv = tk.StringVar(value="8")
		self.var_spatial = tk.IntVar(value=0)
		# Layout
		lbl_mc_cycles = ttk.Label(box_mc, text="MC cycles:")
		lbl_mc_cycles.grid(row=0, column=0, sticky='w', padx=4, pady=2)
		ttk.Entry(box_mc, textvariable=self.var_mc_cycles, width=8).grid(row=0, column=1, sticky='w')
		lbl_sensor_div = ttk.Label(box_mc, text="Sensor division:")
		lbl_sensor_div.grid(row=0, column=2, sticky='w', padx=8)
		ttk.Entry(box_mc, textvariable=self.var_sensor_div, width=8).grid(row=0, column=3, sticky='w')
		lbl_noise_std = ttk.Label(box_mc, text="Noise std (uV):")
		lbl_noise_std.grid(row=0, column=4, sticky='w', padx=8)
		ttk.Entry(box_mc, textvariable=self.var_noise_std, width=8).grid(row=0, column=5, sticky='w')
		lbl_nbasis = ttk.Label(box_mc, text="nbasis:")
		lbl_nbasis.grid(row=1, column=0, sticky='w', padx=4, pady=2)
		ttk.Entry(box_mc, textvariable=self.var_nbasis, width=8).grid(row=1, column=1, sticky='w')
		lbl_l1 = ttk.Label(box_mc, text="l1:")
		lbl_l1.grid(row=1, column=2, sticky='w', padx=8)
		ttk.Entry(box_mc, textvariable=self.var_l1, width=8).grid(row=1, column=3, sticky='w')
		lbl_rep_per_max = ttk.Label(box_mc, text="rep per max:")
		lbl_rep_per_max.grid(row=1, column=4, sticky='w', padx=8)
		ttk.Entry(box_mc, textvariable=self.var_rep_per_max, width=8).grid(row=1, column=5, sticky='w')
		lbl_rep_subdiv = ttk.Label(box_mc, text="rep subdiv:")
		lbl_rep_subdiv.grid(row=2, column=0, sticky='w', padx=4, pady=2)
		ttk.Entry(box_mc, textvariable=self.var_rep_subdiv, width=8).grid(row=2, column=1, sticky='w')
		self.chk_spatial = ttk.Checkbutton(box_mc, text="(Dev.) Compute spatial (slow)", variable=self.var_spatial)
		self.chk_spatial.grid(row=2, column=2, columnspan=2, sticky='w', padx=8)
		# Tooltips for field names
		add_tooltip(lbl_mc_cycles, "Number of Monte Carlo repetitions per sensor subset size.")
		add_tooltip(lbl_sensor_div, "Number of sensors per subset; This must evenly divide the total number of sensors.")
		add_tooltip(lbl_noise_std, "Gaussian noise standard deviation added to voltages (microvolts). >0 Necessary for simulated datasets.")
		add_tooltip(lbl_nbasis, "Number of PCA components utilized in sparse fitting.")
		add_tooltip(lbl_l1, "L1 regularization strength (sparsity penalty) during coefficient fit.")
		add_tooltip(lbl_rep_per_max, "Generates additional replicates based on max sensor count.")
		add_tooltip(lbl_rep_subdiv, "Maximum number of replicates per subdivision when batching to reduce memory usage.")
		add_tooltip(self.chk_spatial, "(In-development) Compute per-label spatial accuracy; slower and more memory-intensive.")
		# Run button and progress
		self.btn_sepio_run = ttk.Button(box_mc, text="Run SEPIO", command=self._sepio_start_mc)
		self.btn_sepio_run.grid(row=3, column=0, padx=4, pady=6, sticky='w')
		self.lbl_mc_status = ttk.Label(box_mc, text="Idle")
		self.lbl_mc_status.grid(row=3, column=1, padx=8, sticky='w')
		self.sepio_progress = ttk.Progressbar(box_mc, orient='horizontal', length=400, mode='determinate')
		self.sepio_progress.grid(row=4, column=0, columnspan=6, pady=6, sticky='w')

		# SEPIO Results Summary (match size of Voltage Summary)
		box_mc_sum = ttk.LabelFrame(frm, text="SEPIO Results Summary")
		box_mc_sum.pack(fill='x', padx=8, pady=8)
		self.txt_sepio_stats = tk.Text(box_mc_sum, height=10, wrap='word')
		self.txt_sepio_stats.configure(bg='#FFFFFF', relief='sunken', bd=2, highlightthickness=0)
		self.txt_sepio_stats.pack(fill='both', expand=True)
		self.txt_sepio_stats.insert('1.0', 'No SEPIO run yet. Set parameters above and press "Run SEPIO".')
		self.txt_sepio_stats.config(state='disabled')

		# Saved SEPIO results management (bottom section)
		box_saved = ttk.LabelFrame(frm, text="Select SEPIO Result")
		box_saved.pack(fill='x', padx=8, pady=8)
		self.var_sepio_saved = tk.StringVar()
		self.cmb_sepio_saved = ttk.Combobox(box_saved, textvariable=self.var_sepio_saved, values=[], width=80, state='readonly')
		self.cmb_sepio_saved.grid(row=0, column=0, sticky='w', padx=4, pady=4)
		btn_update_saved = ttk.Button(box_saved, text="Update List", command=self._sepio_results_refresh_list)
		btn_update_saved.grid(row=0, column=1, padx=6, pady=4)
		btn_del_saved = ttk.Button(box_saved, text="Delete", command=self._sepio_results_delete_selected)
		btn_del_saved.grid(row=0, column=2, padx=6, pady=4)
		# Clear All button for SEPIO results with two-step confirmation
		btn_clear_all_saved = ttk.Button(box_saved, text="Clear All", command=self._sepio_results_clear_all)
		btn_clear_all_saved.grid(row=0, column=3, padx=6, pady=4)


		self._sepio_refresh_results()
		# Populate saved SEPIO results dropdown
		try:
			self._sepio_results_refresh_list()
		except Exception:
			pass

	def _sepio_refresh_results(self):
		labels = []
		if self.project and self.project.results_history:
			for res in self.project.results_history:
				try:
					labels.append(self._label_for_result(res))
				except Exception:
					labels.append(self._label_base_only(res))
		else:
			labels = []
		self.cmb_sepio['values'] = labels
		if labels:
			self.cmb_sepio.current(0)
			self._sepio_on_select()
		else:
			self.sepio_current_result = None
			self.lbl_sepio_details.configure(text="No results available.")

	def _sepio_on_select(self):
		idx = self.cmb_sepio.current()
		if self.project and self.project.results_history and idx >= 0:
			self.sepio_current_result = self.project.results_history[idx]
			self.lbl_sepio_details.configure(text=self._label_for_result(self.sepio_current_result))
		else:
			self.sepio_current_result = None
			self.lbl_sepio_details.configure(text="No result selected.")

	def _sepio_recompute_voltages(self):
		if not self.project:
			messagebox.showerror("SEPIO", "No project loaded.")
			# Also refresh summary to reflect the state
			try:
				self.txt_sepio_summary.config(state='normal')
				self.txt_sepio_summary.delete('1.0', 'end')
				self.txt_sepio_summary.insert('1.0', 'No project loaded. Cannot compute voltages.')
			finally:
				self.txt_sepio_summary.config(state='disabled')
			return
		res = self.sepio_current_result or self.project.results
		if res is None:
			messagebox.showerror("SEPIO", "No optimization result selected.")
			# Refresh summary with notice
			try:
				self.txt_sepio_summary.config(state='normal')
				self.txt_sepio_summary.delete('1.0', 'end')
				self.txt_sepio_summary.insert('1.0', 'No optimization result selected. Compute voltages unavailable.')
			finally:
				self.txt_sepio_summary.config(state='disabled')
			return
		try:
			V, IC = compute_full_voltage_arrays(self.project, res.best_solution)
			self.sepio_voltage = V
			self.sepio_ic = IC
			# Default sensor division to number of sensors in this voltage case
			try:
				n_sensors = int(V.shape[1]) if isinstance(V, np.ndarray) and V.ndim >= 2 else 0
				if n_sensors > 0:
					self.var_sensor_div.set(str(n_sensors))
			except Exception:
				pass
			self._set_status(f"SEPIO: computed V[{V.shape[0]}x{V.shape[1]}] and IC.")
			self._sepio_update_summary()
		except Exception as e:
			messagebox.showerror("SEPIO", f"Voltage computation failed: {e}")
			# Always refresh summary with error context
			try:
				self.txt_sepio_summary.config(state='normal')
				self.txt_sepio_summary.delete('1.0', 'end')
				self.txt_sepio_summary.insert('1.0', f"Voltage computation failed: {e}")
			finally:
				self.txt_sepio_summary.config(state='disabled')

	def _sepio_update_summary(self):
		try:
			self.txt_sepio_summary.config(state='normal')
			self.txt_sepio_summary.delete('1.0', 'end')
			if self.sepio_voltage is None or not isinstance(self.sepio_voltage, np.ndarray):
				self.txt_sepio_summary.insert('1.0', 'No voltage array available.')
				return
			V = self.sepio_voltage
			absV = np.abs(V)
			if V.ndim >= 2:
				n_src, n_sens = int(V.shape[0]), int(V.shape[1])
			else:
				n_src, n_sens = int(V.shape[0]), 0
			finite_mask = np.isfinite(absV)
			finite_ratio = float(np.sum(finite_mask)) / float(absV.size) if absV.size else 0.0
			# Global stats (convert to microvolts for readability)
			absV_uV = absV * 1e6
			mean_all = float(np.nanmean(absV_uV)) if absV_uV.size else float('nan')
			min_all = float(np.nanmin(absV_uV)) if np.any(np.isfinite(absV_uV)) else float('nan')
			max_all = float(np.nanmax(absV_uV)) if np.any(np.isfinite(absV_uV)) else float('nan')
			p95_all = float(np.nanpercentile(absV_uV, 95)) if np.any(np.isfinite(absV_uV)) else float('nan')
			# Per-axis nanmean(abs(V)) summaries
			mean_per_source = np.nanmean(absV_uV, axis=1) if n_sens > 0 else np.array([])
			mean_per_sensor = np.nanmean(absV_uV, axis=0) if n_src > 0 else np.array([])
			def _summ(arr):
				if arr.size == 0 or not np.any(np.isfinite(arr)):
					return {'min': np.nan, 'median': np.nan, 'max': np.nan}
				return {
					'min': float(np.nanmin(arr)),
					'median': float(np.nanmedian(arr)),
					'max': float(np.nanmax(arr))
				}
			src_summ = _summ(mean_per_source)
			sens_summ = _summ(mean_per_sensor)
			# Peak distributions (max over the other axis)
			max_per_source = np.nanmax(absV_uV, axis=1) if n_sens > 0 else np.array([])
			max_per_sensor = np.nanmax(absV_uV, axis=0) if n_src > 0 else np.array([])
			p95_src = float(np.nanpercentile(max_per_source, 95)) if max_per_source.size and np.any(np.isfinite(max_per_source)) else float('nan')
			p95_sens = float(np.nanpercentile(max_per_sensor, 95)) if max_per_sensor.size and np.any(np.isfinite(max_per_sensor)) else float('nan')
			lines = []
			lines.append(f"Voltage array: shape = ({n_src}, {n_sens}) [sources, sensors]; dtype={V.dtype}, ndim={V.ndim}")
			lines.append(f"Finite ratio: {finite_ratio*100:.2f}% ({np.sum(finite_mask)}/{absV.size}) | NaNs={int(np.isnan(absV).sum())}, Infs={int(np.isinf(V).sum())}")
			lines.append(f"Global |V| (µV): mean={mean_all:.3g}, min={min_all:.3g}, max={max_all:.3g}, p95={p95_all:.3g}")
			lines.append("nanmean(|V|) per source (µV): min={min:.3g}, median={median:.3g}, max={max:.3g}".format(**src_summ))
			lines.append("nanmean(|V|) per sensor (µV): min={min:.3g}, median={median:.3g}, max={max:.3g}".format(**sens_summ))
			lines.append(f"Max |V| per source p95 (µV): {p95_src:.3g}")
			lines.append(f"Max |V| per sensor p95 (µV): {p95_sens:.3g}")
			# Diagnostics for finite counts per axis (helpful for debugging)
			if n_sens > 0:
				fin_src_counts = np.sum(np.isfinite(V), axis=1)
				lines.append(f"Finite counts per source: min={int(np.min(fin_src_counts))}, median={float(np.median(fin_src_counts)):.1f}, max={int(np.max(fin_src_counts))}")
			if n_src > 0:
				fin_sens_counts = np.sum(np.isfinite(V), axis=0)
				lines.append(f"Finite counts per sensor: min={int(np.min(fin_sens_counts))}, median={float(np.median(fin_sens_counts)):.1f}, max={int(np.max(fin_sens_counts))}")

			# Multi-device breakdown: split concatenated sensors per device type
			try:
				if self.project and isinstance(self.project.device_counts, list) and self.project.leadfield_files:
					# Determine per-type electrode counts by loading leadfields lightweight
					from scripts.modules.leadfield_importer import FieldImporter as _FI  # type: ignore
					per_type_elec = []
					imp = _FI()
					for lf in self.project.leadfield_files:
						imp.load(lf, clear_fields=True)
						arr = np.asarray(imp.fields)
						per_type_elec.append(int(arr.shape[-1]))
					# Build sensor slices
					start = 0
					per_type_slices = []
					for ne in per_type_elec:
						per_type_slices.append((start, start+ne))
						start += ne
					# Only if counts match current V
					if start == n_sens and per_type_slices:
						lines.append("\nPer-device-type sensor breakdown:")
						for t_idx, (s0, s1) in enumerate(per_type_slices):
							Vt = absV_uV[:, s0:s1]
							mt = np.nanmean(Vt) if Vt.size else np.nan
							mx = np.nanmax(Vt) if np.any(np.isfinite(Vt)) else np.nan
							p95t = float(np.nanpercentile(Vt, 95)) if np.any(np.isfinite(Vt)) else float('nan')
							dev_name = self.project.device_names[t_idx] if t_idx < len(getattr(self.project, 'device_names', [])) else f"Type {t_idx+1}"
							lines.append(f"- {dev_name}: sensors={s1-s0}, |V|(µV) mean={mt:.3g}, max={mx:.3g}, p95={p95t:.3g}")
							if isinstance(self.sepio_ic, np.ndarray) and self.sepio_ic.size:
								ICt = self.sepio_ic[:, s0:s1]
								ICm = float(np.nanmean(ICt)) if ICt.size else float('nan')
								lines.append(f"  IC mean={ICm:.3g}")
			except Exception:
				# If breakdown fails, continue with global summary only
				pass
			# IC stats if available
			if isinstance(self.sepio_ic, np.ndarray) and self.sepio_ic.size:
				IC = self.sepio_ic
				IC_mean = float(np.nanmean(IC))
				IC_src = _summ(np.nanmean(IC, axis=1)) if n_sens > 0 else {'min': np.nan, 'median': np.nan, 'max': np.nan}
				IC_sens = _summ(np.nanmean(IC, axis=0)) if n_src > 0 else {'min': np.nan, 'median': np.nan, 'max': np.nan}
				lines.append(f"IC: mean={IC_mean:.3g}")
				lines.append("IC nan-mean per source: min={min:.3g}, median={median:.3g}, max={max:.3g}".format(**IC_src))
				lines.append("IC nan-mean per sensor: min={min:.3g}, median={median:.3g}, max={max:.3g}".format(**IC_sens))
			self.txt_sepio_summary.insert('1.0', "\n".join(lines))
		except Exception as e:
			# Provide detailed diagnostics if anything goes wrong
			try:
				print("[SEPIO] Error summarizing voltage:", repr(e))
				if isinstance(getattr(self, 'sepio_voltage', None), np.ndarray):
					V = self.sepio_voltage
					print(f"[SEPIO] V diagnostics -> dtype={V.dtype}, ndim={V.ndim}, shape={V.shape}, size={V.size}")
					print(f"[SEPIO] isfinite.sum={np.isfinite(V).sum()}, isnan.sum={np.isnan(V).sum()}, isinf.sum={np.isinf(V).sum()}")
			except Exception:
				pass
			self.txt_sepio_summary.insert('1.0', f"Error summarizing voltage: {e}\nPlease check console for diagnostics.")
		finally:
			self.txt_sepio_summary.config(state='disabled')

	def _sepio_plot_voltage_sources(self):
		if not self._matplotlib_required():
			return
		# Ensure plot is only shown on explicit button action; suppress during SEPIO run
		if getattr(self, 'sepio_is_running', False):
			return
		if self.sepio_voltage is None:
			messagebox.showinfo("SEPIO", "Compute voltages first.")
			return
		fig = None
		try:
			arr = np.max(np.abs(self.sepio_voltage), axis=1) * 1e6  # µV
			if arr.size == 0:
				messagebox.showinfo("SEPIO", "No sources available to plot.")
				return
			mask = np.isfinite(arr)
			if not np.any(mask):
				messagebox.showinfo("SEPIO", "All source voltages are NaN.")
				return
			clip_limit = np.percentile(arr[mask], 95)
			if not np.isfinite(clip_limit) or clip_limit <= 0:
				clip_limit = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
			arr = np.clip(arr, 0, clip_limit)
			fig = plt.figure(figsize=(6,4))
			plt.hist(arr[mask], bins=50, color='#446AC2', alpha=0.8)
			plt.xlim(0, clip_limit*1.1)
			plt.title('Voltage Distribution (sources)')
			plt.xlabel('Max |µV| across sensors')
			plt.ylabel('Count')
			png = self._plot_to_temp_png(fig)
			self._show_image_window("SEPIO: Sources Voltage Distribution", png, save_label="Save to Project", save_cb=lambda p: self._save_plot_to_project(p))
		finally:
			try:
				if fig is not None:
					plt.close(fig)
			except Exception:
				pass

	def _sepio_start_mc(self):
		# Validate prerequisites
		if self.sepio_voltage is None or not isinstance(self.sepio_voltage, np.ndarray):
			messagebox.showinfo("SEPIO", "Compute voltages first (Re-compute Voltages).")
			return
		try:
			cycles = max(1, int(float(self.var_mc_cycles.get())))
			sensor_div = max(1, int(float(self.var_sensor_div.get())))
			noise_std = float(self.var_noise_std.get()) * 1e-6  # Convert µV to V
			nbasis = int(float(self.var_nbasis.get()))
			l1 = float(self.var_l1.get())
			rep_per_max = max(1, int(float(self.var_rep_per_max.get())))
			rep_subdiv = max(1, int(float(self.var_rep_subdiv.get())))
			spatial = bool(self.var_spatial.get())
		except Exception:
			messagebox.showerror("SEPIO", "Invalid parameter values. Please check entries.")
			return
		# Enforce sensor_div to be divisions of 2 from the total number of sensors
		try:
			n_sensors = int(self.sepio_voltage.shape[1]) if self.sepio_voltage.ndim >= 2 else 0
			if n_sensors <= 0:
				messagebox.showerror("SEPIO", "Invalid voltage shape: zero sensors.")
				return
			allowed = [1]
			divisors = [2,3,5,7,11] # primes to use as divisors to find allowed sensor subsets
			for divisor in divisors:
				val = n_sensors
				while val >= 1:
					if val not in allowed:
						allowed.append(int(val))
					if val % divisor != 0:
						break
					val //= divisor
			if sensor_div not in allowed:
				self.var_sensor_div.set(str(n_sensors))
				messagebox.showerror("SEPIO", f"Sensor division must be one of {allowed} for total {n_sensors} sensors.")
				return
		except Exception:
			# If validation fails silently, continue; sepio may still handle
			pass
		# Prepare progress UI
		# Progress reflects single mc_train call; cycles passed to MCcount only
		self.sepio_progress['maximum'] = 1
		self.sepio_progress['value'] = 0
		self.lbl_mc_status.configure(text=f"Running...")
		self.btn_sepio_run.configure(state='disabled')
		# Guard against unintended plotting during run
		self.sepio_is_running = True
		# Clear summary
		self._sepio_clear_stats()
		# Launch worker thread
		import threading
		args = {
			'cycles': cycles,
			'sensor_div': sensor_div,
			'noise_std': noise_std,
			'nbasis': nbasis,
			'l1': l1,
			'rep_per_max': rep_per_max,
			'rep_subdiv': rep_subdiv,
			'spatial': spatial,
		}
		threading.Thread(target=self._sepio_run_mc_worker, args=(args,), daemon=True).start()

	def _sepio_run_mc_worker(self, args: dict):
		from scripts.modules.SEPIO import sepio
		try:
			X = np.nan_to_num(self.sepio_voltage)
			cycles = args['cycles']
			sensor_div = args['sensor_div']
			noise_std = args['noise_std']
			nbasis = args['nbasis']
			l1 = args['l1']
			rep_per_max = args['rep_per_max']
			rep_subdiv = args['rep_subdiv']
			spatial = args['spatial']
			# Single call: pass cycles into MCcount, do not loop/average locally
			MCcoefs, MCaccs, MCSaccs, sensor_range = sepio.mc_train(
				X=X, y=None, Xt=None, yt=None,
				sensor_div=sensor_div,
				MCcount=cycles,
				noise=noise_std,
				replicates=rep_per_max*X.shape[-1],
				nbasis=nbasis,
				spatial=spatial,
				l1=l1,
				rep_subdiv=rep_subdiv
			)
			# Update progress (single step)
			self.after(0, lambda: self._sepio_update_progress(1))
			self.after(0, lambda: self._sepio_finish_mc(MCcoefs, MCaccs, MCSaccs if isinstance(MCSaccs, np.ndarray) else None, sensor_range))
		except Exception as e:
			self.after(0, lambda: messagebox.showerror("SEPIO", f"SEPIO run failed: {e}"))
			self.after(0, lambda: self._sepio_reset_progress())

	def _sepio_update_progress(self, value: int):
		self.sepio_progress['value'] = value
		self.lbl_mc_status.configure(text=f"Completed {value}/{int(self.sepio_progress['maximum'])}")

	def _sepio_finish_mc(self, coefs_avg: np.ndarray, accs_avg: np.ndarray, saccs_avg: Optional[np.ndarray], sensor_range: np.ndarray):
		self._set_status("SEPIO: Monte-Carlo run finished.")
		self.lbl_mc_status.configure(text="Done")
		self.btn_sepio_run.configure(state='normal')
		self.sepio_is_running = False
		# Store
		self.sepio_mc_coefs = coefs_avg
		self.sepio_mc_accs = accs_avg
		self.sepio_mc_saccs = saccs_avg
		self.sepio_sensor_range = sensor_range
		# Persist into project SEPIO results list with labeling
		try:
			self._store_sepio_result(coefs_avg, accs_avg, saccs_avg, sensor_range)
		except Exception as e:
			print("[SEPIO] Warning: failed to persist SEPIO result:", e)
		# Update stats panel
		self._sepio_update_stats()
		# Refresh saved-list UI
		try:
			self._sepio_results_refresh_list()
		except Exception:
			pass
		# Ensure Plotting tab sees the new SEPIO result immediately
		try:
			self._plot_refresh_dropdown()
		except Exception:
			pass

	def _sepio_reset_progress(self):
		self.lbl_mc_status.configure(text="Idle")
		self.sepio_progress['value'] = 0
		self.btn_sepio_run.configure(state='normal')

	def _sepio_clear_stats(self):
		self.txt_sepio_stats.config(state='normal')
		self.txt_sepio_stats.delete('1.0', 'end')
		self.txt_sepio_stats.insert('1.0', 'Running SEPIO...')
		self.txt_sepio_stats.config(state='disabled')

	def _sepio_update_stats(self):
		try:
			self.txt_sepio_stats.config(state='normal')
			self.txt_sepio_stats.delete('1.0', 'end')
			if not isinstance(getattr(self, 'sepio_mc_accs', None), np.ndarray) or getattr(self, 'sepio_sensor_range', None) is None:
				self.txt_sepio_stats.insert('1.0', 'No SEPIO results available.')
				return
			accs = self.sepio_mc_accs
			sensors = self.sepio_sensor_range
			best_idx = int(np.nanargmax(accs)) if accs.size else 0
			best_acc = float(accs[best_idx]) if accs.size else float('nan')
			best_sens = int(sensors[best_idx]) if sensors.size else 0
			mean_acc = float(np.nanmean(accs)) if accs.size else float('nan')
			lines = []
			lines.append(f"Sensor range tested: {list(map(int, sensors.tolist()))}")
			lines.append(f"Accuracy (sensor space): mean={mean_acc:.3f}, best={best_acc:.3f} at {best_sens} sensors")
			# Coefficients summary (top sensors)
			coefs = getattr(self, 'sepio_mc_coefs', None)
			if isinstance(coefs, np.ndarray) and coefs.size:
				if coefs.ndim == 1:
					weights_raw = np.abs(coefs)
				else:
					weights_raw = np.abs(coefs).sum(axis=1)
				# Determine ranking based on raw weights
				order = np.argsort(-weights_raw)
				top_k = order[:10].tolist()
				# Normalize weights across the array for display
				max_w = float(np.nanmax(weights_raw)) if weights_raw.size else 0.0
				if np.isfinite(max_w) and max_w > 0:
					weights = weights_raw / max_w
				else:
					weights = np.zeros_like(weights_raw)
				lines.append(f"Top sensors by importance: {top_k}")
				lines.append(f"Top weights (norm): {[float(weights[i]) for i in top_k]}")
			# Spatial acc summary if available
			saccs = getattr(self, 'sepio_mc_saccs', None)
			if isinstance(saccs, np.ndarray) and saccs.size and (np.nansum(saccs) > 1e-6):
				lines.append(f"Spatial accuracy per label: {[float(v) for v in saccs.tolist()]}")
			self.txt_sepio_stats.insert('1.0', "\n".join(lines))
		finally:
			self.txt_sepio_stats.config(state='disabled')

	def _sepio_plot_voltage_sensors(self):
		if not self._matplotlib_required():
			return
		if self.sepio_voltage is None:
			messagebox.showinfo("SEPIO", "Compute voltages first.")
			return
		fig = None
		try:
			arr = np.max(np.abs(self.sepio_voltage), axis=0) * 1e6  # µV
			if arr.size == 0:
				messagebox.showinfo("SEPIO", "No sensors available to plot.")
				return
			mask = np.isfinite(arr)
			if not np.any(mask):
				messagebox.showinfo("SEPIO", "All sensor voltages are NaN.")
				return
			clip_limit = np.percentile(arr[mask], 95)
			if not np.isfinite(clip_limit) or clip_limit <= 0:
				clip_limit = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
			arr = np.clip(arr, 0, clip_limit)
			fig = plt.figure(figsize=(6,4))
			plt.hist(arr[mask], bins=50, color='#C14343', alpha=0.8)
			plt.xlim(0, clip_limit*1.1)
			plt.title('Voltage Distribution (sensors)')
			plt.xlabel('Max |µV| across sources')
			plt.ylabel('Count')
			png = self._plot_to_temp_png(fig)
			self._show_image_window("SEPIO: Sensors Voltage Distribution", png, save_label="Save to Project", save_cb=lambda p: self._save_plot_to_project(p))
		finally:
			try:
				if fig is not None:
					plt.close(fig)
			except Exception:
				pass

	# --------------------------------------------------------------
	# Menubar & Status helpers
	# --------------------------------------------------------------
	def _build_menubar(self):
		menubar = tk.Menu(self)
		file_menu = tk.Menu(menubar, tearoff=0)
		file_menu.add_command(label='New', command=self._clear_form)
		file_menu.add_command(label='Load...', command=self._load_project)
		file_menu.add_command(label='Save', command=self._save_project)
		file_menu.add_separator()
		file_menu.add_command(label='Exit', command=self.destroy)
		menubar.add_cascade(label='File', menu=file_menu)

		proj_menu = tk.Menu(menubar, tearoff=0)
		proj_menu.add_command(label='Create / Update Project', command=self._create_or_update_project)
		menubar.add_cascade(label='Project', menu=proj_menu)

		help_menu = tk.Menu(menubar, tearoff=0)
		help_menu.add_command(label='About', command=self._show_about)
		menubar.add_cascade(label='Help', menu=help_menu)

		self.config(menu=menubar)

	def _set_status(self, text: str):
		try:
			self.status_var.set(text)
		except Exception:
			pass

	# --------------------------------------------------------------
	# SEPIO saved results helpers
	# --------------------------------------------------------------
	def _store_sepio_result(self, coefs_avg: np.ndarray, accs_avg: np.ndarray, saccs_avg: Optional[np.ndarray], sensor_range: np.ndarray):
		"""Create a labeled SEPIOResult entry and append to project, then save.

		Label includes optimization result title, number of devices, and SEPIO settings.
		"""
		proj = getattr(self, 'project', None)
		if proj is None:
			return
		# Determine linked optimization result and label
		res = self.sepio_current_result or getattr(proj, 'results', None)
		if res is None:
			return
		opt_label = self._label_for_result(res)
		n_devices = int((res.best_solution.size // 6) if getattr(res, 'best_solution', None) is not None else np.sum(getattr(proj, 'device_counts', []) or [0]))
		# Collect SEPIO settings from UI controls
		try:
			cycles = max(1, int(float(self.var_mc_cycles.get())))
			sensor_div = max(1, int(float(self.var_sensor_div.get())))
			noise_std = float(self.var_noise_std.get())
			nbasis = int(float(self.var_nbasis.get()))
			l1 = float(self.var_l1.get())
			rep_per_max = max(1, int(float(self.var_rep_per_max.get())))
			rep_subdiv = max(1, int(float(self.var_rep_subdiv.get())))
			spatial = bool(self.var_spatial.get())
		except Exception:
			# Fallback defaults if UI parsing fails
			cycles = 1; sensor_div = 16; noise_std = 0.01; nbasis = 20; l1 = 0.001; rep_per_max = 1; rep_subdiv = 8; spatial = False
		settings = {
			'cycles': cycles,
			'sensor_div': sensor_div,
			'noise_std': noise_std,
			'nbasis': nbasis,
			'l1': l1,
			'rep_per_max': rep_per_max,
			'rep_subdiv': rep_subdiv,
			'spatial': spatial,
		}
		# Build base label and ensure uniqueness
		base = f"{opt_label}-N{n_devices}-Div{sensor_div}-Noise{noise_std}-nb{nbasis}-l1{l1}-Rep{rep_per_max}x{rep_subdiv}-Cyc{cycles}"
		labels_in_use = set()
		for entry in getattr(proj, 'sepio_results', []) or []:
			try:
				lab = entry.label if hasattr(entry, 'label') else entry.get('label')
				if isinstance(lab, str):
					labels_in_use.add(lab)
			except Exception:
				pass
		label = base
		if label in labels_in_use:
			idx = 1
			while True:
				cand = f"{base}-rerun{idx}"
				if cand not in labels_in_use:
					label = cand
					break
				idx += 1
		# Create and append dataclass entry
		try:
			entry = SEPIOResult(
				label=label,
				opt_label=opt_label,
				n_devices=n_devices,
				settings=settings,
				coefs_avg=np.asarray(coefs_avg) if coefs_avg is not None else np.zeros(0),
				accs_avg=np.asarray(accs_avg) if accs_avg is not None else np.zeros(0),
				saccs_avg=(np.asarray(saccs_avg) if saccs_avg is not None else None),
				sensor_range=np.asarray(sensor_range) if sensor_range is not None else np.zeros(0, dtype=int),
			)
		except Exception as e:
			print("[SEPIO] Failed building SEPIOResult:", e)
			return
		if not hasattr(proj, 'sepio_results') or proj.sepio_results is None:
			proj.sepio_results = []
		proj.sepio_results.append(entry)
		# Save project silently if possible
		try:
			loaded_path = getattr(proj, 'loaded_path', None)
			if loaded_path:
				proj.save(loaded_path)
		except Exception:
			pass

	def _sepio_results_refresh_list(self):
		items = []
		self._sepio_saved_objects = []
		proj = getattr(self, 'project', None)
		if proj and getattr(proj, 'sepio_results', None):
			for ent in proj.sepio_results:
				lab = ent.label if hasattr(ent, 'label') else ent.get('label', '')
				items.append(lab)
				self._sepio_saved_objects.append(ent)
		self.cmb_sepio_saved['values'] = items
		if items:
			self.var_sepio_saved.set(items[-1])
		else:
			self.var_sepio_saved.set('')
		# Keep Plotting tab's dropdown in sync with SEPIO results
		try:
			self._plot_refresh_dropdown()
		except Exception:
			pass

	def _sepio_results_delete_selected(self):
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		label = self.var_sepio_saved.get()
		if not label:
			messagebox.showinfo("Delete", "No SEPIO result selected.")
			return
		confirm = messagebox.askyesno("Confirm Deletion", f"Delete SEPIO result '{label}' permanently?")
		if not confirm:
			return
		try:
			new_list = []
			for ent in getattr(proj, 'sepio_results', []) or []:
				lab = ent.label if hasattr(ent, 'label') else ent.get('label')
				if lab != label:
					new_list.append(ent)
			proj.sepio_results = new_list
			# Persist if a path is known
			loaded_path = getattr(proj, 'loaded_path', None)
			if loaded_path:
				proj.save(loaded_path)
			self._sepio_results_refresh_list()
			# Also refresh Plotting dropdown after deletion
			try:
				self._plot_refresh_dropdown()
			except Exception:
				pass
		except Exception as e:
			messagebox.showerror("Delete Error", f"Failed to remove SEPIO result: {e}")

	def _sepio_results_clear_all(self):
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		# Check if there are any SEPIO results
		if not getattr(proj, 'sepio_results', None):
			messagebox.showinfo("Clear All", "No SEPIO results to clear.")
			return
		# First confirmation
		ok1 = messagebox.askyesno(
			"Clear All SEPIO Results",
			"This will delete ALL saved SEPIO results for this project. Continue?"
		)
		if not ok1:
			return
		# Second confirmation
		ok2 = messagebox.askyesno(
			"Confirm Permanent Deletion",
			"Are you absolutely sure? This action cannot be undone."
		)
		if not ok2:
			return
		try:
			proj.sepio_results = []
			# Persist if a path is known
			loaded_path = getattr(proj, 'loaded_path', None)
			if loaded_path:
				proj.save(loaded_path)
			# Refresh UI
			self._sepio_results_refresh_list()
			# Also refresh Plotting dropdown after clearing
			try:
				self._plot_refresh_dropdown()
			except Exception:
				pass
			messagebox.showinfo("Clear All", "All SEPIO results have been cleared.")
		except Exception as e:
			messagebox.showerror("Clear All Error", f"Failed to clear SEPIO results: {e}")

	def _show_about(self):
		messagebox.showinfo(
			"About",
			"MDPO Unified Interface\nTranslational Bioelectronics Lab"
		)

	# --------------------------------------------------------------
	# Helper: Browsing & Form actions
	# --------------------------------------------------------------
	def _browse_dir(self, var: tk.StringVar):
		d = filedialog.askdirectory()
		if d:
			var.set(d)

	def _browse_file(self, var: tk.StringVar, ext=("All","*.*")):
		f = filedialog.askopenfilename(filetypes=[ext])
		if f:
			var.set(f)

	def _add_roi(self):
		f = filedialog.askopenfilename(filetypes=[("MAT files","*.mat")])
		if f:
			self.lb_roi.insert('end', f)

	def _remove_roi(self):
		sel = list(self.lb_roi.curselection())
		sel.reverse()
		for i in sel:
			self.lb_roi.delete(i)

	def _load_struct_mat(self, path: str) -> Optional[np.ndarray]:
		"""Load a 4xN or Nx4 MAT array and return as (M,4) [x,y,z,r] in recentered space.

		Select the first non-meta key that matches shape containing a 4-length axis.
		"""
		try:
			from scipy.io import loadmat
			d = loadmat(path)
			keys = [k for k in d.keys() if not str(k).startswith('__')]
			arr = None
			for k in keys:
				v = d[k]
				if hasattr(v, 'shape') and len(v.shape) >= 2:
					sh = v.shape
					if 4 in sh:
						arr = np.asarray(v, dtype=float)
						break
			if arr is None:
				messagebox.showwarning("Invalid MAT", f"No 4xN array found in: {os.path.basename(path)}")
				return None
			# Normalize to (M,4) with columns [x,y,z,r]
			if arr.shape[0] == 4:
				arr = arr.T
			elif arr.shape[-1] == 4:
				arr = arr.reshape((-1, 4))
			else:
				# Fallback: attempt squeeze and check again
				arr = np.squeeze(arr)
				if arr.ndim == 2 and (arr.shape[0] == 4 or arr.shape[1] == 4):
					arr = arr.T if arr.shape[0] == 4 else arr
				else:
					messagebox.showwarning("Invalid Shape", f"Could not normalize shape for: {os.path.basename(path)}")
					return None
			# Ensure finite numbers; drop rows with NaNs
			arr = np.asarray(arr, dtype=float)
			mask = ~np.isnan(arr).any(axis=1)
			arr = arr[mask]
			return arr
		except Exception as e:
			messagebox.showerror("Load Error", f"Failed to load {os.path.basename(path)}: {e}")
			return None

	def _add_struct(self):
		f = filedialog.askopenfilename(filetypes=[("MAT files","*.mat")])
		if f:
			self.lb_struct.insert('end', f)

	def _remove_struct(self):
		sel = list(getattr(self, 'lb_struct', tk.Listbox()).curselection())
		sel.reverse()
		for i in sel:
			self.lb_struct.delete(i)

	def _compute_structural_avoidance(self) -> np.ndarray:
		"""Aggregate avoidance spheres from selected files and optional ROI demo.

		Returns an array (K,4) with rows [x,y,z,r] in recentered brain space.
		"""
		rows: list = []
		# From provided .mat files
		try:
			paths = list(self.lb_struct.get(0, 'end')) if hasattr(self, 'lb_struct') else []
			for pth in paths:
				arr = self._load_struct_mat(pth)
				if arr is not None and arr.size > 0:
					rows.append(arr)
		except Exception:
			pass
		# Demo @ ROI: use center of each ROI with provided radius
		try:
			if bool(self.var_struct_demo.get()):
				rad = float(self.var_struct_radius.get()) if self.var_struct_radius.get() else 5.0
				p = self.project
				if p and p.brain_vertices is not None:
					brain_center = np.mean(np.asarray(p.brain_vertices), axis=0)
				else:
					brain_center = None
				roi_paths = list(self.lb_roi.get(0, 'end')) if hasattr(self, 'lb_roi') else []
				for rp in roi_paths:
					try:
						from GUI.mdpo_project import parse_mesh_from_mat
						mesh = parse_mesh_from_mat(rp)
						verts = np.asarray(mesh['vertices'], dtype=float)
						if brain_center is not None:
							verts = verts - brain_center
						cent = np.mean(verts, axis=0)
						rows.append(np.array([[cent[0], cent[1], cent[2], rad]], dtype=float))
					except Exception:
						# Skip malformed ROI
						continue
		except Exception:
			pass
		out = np.vstack(rows) if rows else np.zeros((0,4), dtype=float)
		self.structural_avoidance = out
		# Also store on project if available for downstream use
		if getattr(self, 'project', None) is not None:
			try:
				setattr(self.project, 'structural_avoidance', out)
			except Exception:
				pass
		return out

	def _add_lf(self):
		f = filedialog.askopenfilename(filetypes=[("NumPy .npz","*.npz")])
		if f:
			# Insert with Type N label
			idx = len(self.tv_lf.get_children()) + 1
			self.tv_lf.insert('', 'end', values=(f"Type {idx}", f))
			self._rebuild_device_grid()

	def _remove_lf(self):
		# Update for treeview selection
		items = self.tv_lf.selection()
		for it in items:
			self.tv_lf.delete(it)
		# Re-label remaining rows to maintain Type order
		for idx, it in enumerate(self.tv_lf.get_children(), start=1):
			vals = self.tv_lf.item(it, 'values')
			self.tv_lf.item(it, values=(f"Type {idx}", vals[1]))
		self._rebuild_device_grid()

	def _clear_form(self):
		self.var_name.set("Untitled")
		self.var_folder.set("")
		self.var_brain.set("")
		self.lb_roi.delete(0,'end')
		# Clear structural avoidance list and reset options
		if hasattr(self, 'lb_struct'):
			self.lb_struct.delete(0,'end')
		self.var_struct_demo.set(False)
		self.var_struct_radius.set("5.0")
		# Clear leadfields tree
		for it in self.tv_lf.get_children():
			self.tv_lf.delete(it)
		# Clear device grid
		for arr_list in [self.dev_name_entries,self.dev_count_entries,self.dev_noise_entries,self.dev_bw_entries,self.dev_scale_entries,self.dev_angle_entries,self.dev_depth_min_entries,self.dev_depth_max_entries,self.dev_prox_entries,self.dev_length_entries,self.dev_radius_entries]:
			for ent in arr_list:
				ent.destroy()
			arr_list.clear()
		for w in self.devices_grid.winfo_children():
			w.destroy()
		self.lbl_devices_status.config(text="Cleared")
		self.var_measure.set("ic")
		self.var_gens.set("30")
		self.var_pop.set("44")
		self.var_parents.set("15")
		self.var_proc.set("4")
		self.lbl_status_setup.config(text="Form cleared")

	# Legacy no-op maintained for backward compatibility
	def _clear_constraint_fields(self):
		pass

	def _rebuild_device_grid(self):
		# Build columns equal to number of leadfield files
		for w in self.devices_grid.winfo_children():
			w.destroy()
		# Reset entry lists
		for arr in [self.dev_name_entries,self.dev_count_entries,self.dev_noise_entries,self.dev_bw_entries,self.dev_scale_entries,self.dev_angle_entries,self.dev_depth_min_entries,self.dev_depth_max_entries,self.dev_prox_entries,self.dev_length_entries,self.dev_radius_entries]:
			arr.clear()
		# Get leadfield paths in Type order
		lf_files = [self.tv_lf.item(it,'values')[1] for it in self.tv_lf.get_children()]
		n_types = len(lf_files)
		if n_types == 0:
			self.lbl_devices_status.config(text="No leadfields")
			return
		# Header row with device attributes
		headers = ["Device Name","# Devices","STD Noise (uV)","Bandwidth (Bits/s)","Voxel Scale (mm)","Angle limit (deg)","Depth min (mm)","Depth max (mm)","Proximity Min (mm)","Device Length (mm)","Device Radius (mm)"]
		for c in range(n_types):
			col_frame = ttk.LabelFrame(self.devices_grid, text=f"Type {c+1}")
			col_frame.grid(row=0, column=c, padx=6, pady=4, sticky='n')
			for r, label in enumerate(headers):
				ttk.Label(col_frame, text=label+':').grid(row=r, column=0, sticky='w')
				ent = ttk.Entry(col_frame, width=14)
				ent.grid(row=r, column=1, sticky='w')
				# Store in corresponding list
				if r == 0: self.dev_name_entries.append(ent)
				elif r == 1: self.dev_count_entries.append(ent)
				elif r == 2: self.dev_noise_entries.append(ent)
				elif r == 3: self.dev_bw_entries.append(ent)
				elif r == 4: self.dev_scale_entries.append(ent)
				elif r == 5: self.dev_angle_entries.append(ent)
				elif r == 6: self.dev_depth_min_entries.append(ent)
				elif r == 7: self.dev_depth_max_entries.append(ent)
				elif r == 8: self.dev_prox_entries.append(ent)
				elif r == 9: self.dev_length_entries.append(ent)
				elif r == 10: self.dev_radius_entries.append(ent)
		self.lbl_devices_status.config(text=f"{n_types} device types configured")

	# --------------------------------------------------------------
	# Optimization parameter selection (dynamic)
	# --------------------------------------------------------------
	def _rebuild_param_selection(self):
		for w in self.param_box.winfo_children():
			w.destroy()
		# First row: common across all methods
		common = ttk.Frame(self.param_box)
		common.pack(fill='x', padx=6, pady=6)
		lbl_init = tk.Label(common, text="Initial Solutions:")
		lbl_init.grid(row=0, column=0, sticky='w')
		add_tooltip(lbl_init, "Number of seed solutions; used for all methods")
		tk.Entry(common, textvariable=self.var_init_solutions, width=12).grid(row=0, column=1, sticky='w')
		method = self.var_run_method.get()
		if method == 'anneal':
			container = ttk.Frame(self.param_box)
			container.pack(fill='x', padx=6, pady=6)
			labels = ["Iterations","Initial Temp","Final Temp","Cooling Rate","Cartesian Step","Rotational Step"]
			vars = [self.var_a_iter,self.var_a_itemp,self.var_a_ftemp,self.var_a_cool,self.var_a_cart,self.var_a_rot]
			tooltips = {
				"Iterations": "Inner iterations per temperature level",
				"Initial Temp": "Starting temperature for annealing",
				"Final Temp": "Stopping temperature threshold",
				"Cooling Rate": "Temperature decay factor per step (0-1)",
				"Cartesian Step": "Position perturbation scale (mm)",
				"Rotational Step": "Angle perturbation scale (radians)"
			}
			for i,(lab,var_) in enumerate(zip(labels,vars)):
				row = i//3; col = (i%3)*2
				lbl = tk.Label(container, text=lab+':')
				lbl.grid(row=row, column=col, sticky='w')
				add_tooltip(lbl, tooltips.get(lab, lab))
				tk.Entry(container, textvariable=var_, width=12).grid(row=row, column=col+1, sticky='w')
		elif method == 'multianneal':
			container = ttk.Frame(self.param_box)
			container.pack(fill='x', padx=6, pady=6)
			labels = ["Iterations","Initial Temp","Final Temp","Cooling Rate","Cartesian Step","Rotational Step","Restarts"]
			vars = [self.var_a_iter,self.var_a_itemp,self.var_a_ftemp,self.var_a_cool,self.var_a_cart,self.var_a_rot,self.var_ma_restarts]
			tooltips = {
				"Iterations": "Inner iterations per temperature level",
				"Initial Temp": "Starting temperature for annealing",
				"Final Temp": "Stopping temperature threshold",
				"Cooling Rate": "Temperature decay factor per step (0-1)",
				"Cartesian Step": "Position perturbation scale (mm)",
				"Rotational Step": "Angle perturbation scale (radians)",
				"Restarts": "Number of independent anneal runs"
			}
			for i,(lab,var_) in enumerate(zip(labels,vars)):
				row = i//3; col = (i%3)*2
				lbl = tk.Label(container, text=lab+':')
				lbl.grid(row=row, column=col, sticky='w')
				add_tooltip(lbl, tooltips.get(lab, lab))
				tk.Entry(container, textvariable=var_, width=12).grid(row=row, column=col+1, sticky='w')
		elif method == 'mSGD':
			container = ttk.Frame(self.param_box)
			container.pack(fill='x', padx=6, pady=6)
			labels = ["Iterations","Instances","Top-K","Angle Step","Cart Step","Threshold","Decay"]
			vars = [self.var_bb_iter,self.var_bb_instances,self.var_bb_top,self.var_bb_angle,self.var_bb_cart,self.var_bb_thresh,self.var_bb_decay]
			tooltips = {
				"Iterations": "Number of zoom iterations",
				"Instances": "Random proposals per iteration",
				"Top-K": "Best proposals to keep each iteration",
				"Angle Step": "Angular perturbation magnitude (radians)",
				"Cart Step": "Positional perturbation magnitude (mm)",
				"Threshold": "Early-stop improvement threshold",
				"Decay": "Step decay factor per iteration"
			}
			for i,(lab,var_) in enumerate(zip(labels,vars)):
				row = i//3; col = (i%3)*2
				lbl = tk.Label(container, text=lab+':')
				lbl.grid(row=row, column=col, sticky='w')
				add_tooltip(lbl, tooltips.get(lab, lab))
				tk.Entry(container, textvariable=var_, width=12).grid(row=row, column=col+1, sticky='w')
		elif method == 'brute':
			container = ttk.Frame(self.param_box)
			container.pack(fill='x', padx=6, pady=6)
			labels = ["Limit","Batch Size"]
			vars = [self.var_bf_limit,self.var_bf_batch]
			tooltips = {
				"Limit": "Max number of initial solutions to evaluate",
				"Batch Size": "Solutions to score per iteration"
			}
			for i,(lab,var_) in enumerate(zip(labels,vars)):
				row = 0; col = i*2
				lbl = tk.Label(container, text=lab+':')
				lbl.grid(row=row, column=col, sticky='w')
				add_tooltip(lbl, tooltips.get(lab, lab))
				tk.Entry(container, textvariable=var_, width=12).grid(row=row, column=col+1, sticky='w')
		else:
			container = ttk.Frame(self.param_box)
			container.pack(fill='x', padx=6, pady=6)
			# Genetic parameters
			labels = ["Generations","Pop Size","Parents Mating","Processes","Mutation Prob (0-1)"]
			vars = [self.var_gens,self.var_pop,self.var_parents,self.var_proc,self.var_mut_prob]
			tooltips = {
				"Generations": "Number of GA generations",
				"Pop Size": "Individuals per generation",
				"Parents Mating": "Parents used to produce offspring",
				"Processes": "Threads used for scoring",
				"Mutation Prob (0-1)": "Per-gene mutation probability"
			}
			for i,(lab,var_) in enumerate(zip(labels,vars)):
				row = i//3; col = (i%3)*2
				lbl = tk.Label(container, text=lab+':')
				lbl.grid(row=row, column=col, sticky='w')
				add_tooltip(lbl, tooltips.get(lab, lab))
				tk.Entry(container, textvariable=var_, width=12).grid(row=row, column=col+1, sticky='w')

	# --------------------------------------------------------------
	# Project creation, save, load
	# --------------------------------------------------------------
	def _create_or_update_project(self):
		if Project is None:
			messagebox.showerror("Import Error", "mdpo_project module not available.")
			return
		try:
			# Align device columns to current leadfield count
			lf_count = len(self.tv_lf.get_children())
			if lf_count == 0:
				raise ValueError("Add at least one lead field to configure devices.")
			if len(self.dev_name_entries) != lf_count:
				self._rebuild_device_grid()
			# Collect per-type values
			names: list[str] = []
			counts: list[int] = []
			noise: list[float] = []
			bw: list[float] = []
			scale: list[float] = []
			angle_deg: list[float] = []
			depth_limits: list[tuple[float,float]] = []
			prox_limits: list[float] = []
			device_lengths: list[float] = []  # init physical dimension lists
			device_radii: list[float] = []
			for i in range(lf_count):
				names.append((self.dev_name_entries[i].get() or "Device").strip())
				def _f(ent, default):
					val = ent.get().strip()
					try:
						return float(val)
					except Exception:
						return default
				def _i(ent, default):
					val = ent.get().strip()
					try:
						return int(val)
					except Exception:
						return default
				counts.append(_i(self.dev_count_entries[i], 1))
				noise.append(_f(self.dev_noise_entries[i], 2.3))
				bw.append(_f(self.dev_bw_entries[i], 100.0))
				scale.append(_f(self.dev_scale_entries[i], 0.5))
				angle_deg.append(_f(self.dev_angle_entries[i], 0.0))
				dmin = self.dev_depth_min_entries[i].get().strip()
				dmax = self.dev_depth_max_entries[i].get().strip()
				depth_limits.append((float(dmin) if dmin else np.nan, float(dmax) if dmax else np.nan))
				pmin = self.dev_prox_entries[i].get().strip()
				prox_limits.append(float(pmin) if pmin else (self.project.cl_offset if self.project else 3.0))
				# Physical dimensions
				len_raw = self.dev_length_entries[i].get().strip() if i < len(self.dev_length_entries) else ''
				rad_raw = self.dev_radius_entries[i].get().strip() if i < len(self.dev_radius_entries) else ''
				try:
					len_val = float(len_raw) if len_raw else 10.0
					if len_val <= 0: len_val = 10.0
				except Exception:
					len_val = 10.0
				try:
					rad_val = float(rad_raw) if rad_raw else 0.5
					if rad_val <= 0: rad_val = 0.5
				except Exception:
					rad_val = 0.5
				device_lengths.append(len_val)
				device_radii.append(rad_val)
			# Validate
			for idx,a in enumerate(angle_deg):
				if a < 0:
					raise ValueError(f"Angle limit cannot be negative (device type {idx+1}).")
			for idx,(mn,mx) in enumerate(depth_limits):
				if not np.isnan(mn) and not np.isnan(mx) and mn > mx:
					raise ValueError(f"Depth min exceeds max for device type {idx+1}.")
			for idx,pl in enumerate(prox_limits):
				if pl < 0:
					raise ValueError(f"Proximity limit cannot be negative (device type {idx+1}).")
			angle_rad = [a*np.pi/180.0 for a in angle_deg]
			roi_files = list(self.lb_roi.get(0,'end'))
			lf_files = [self.tv_lf.item(it,'values')[1] for it in self.tv_lf.get_children()]
			roi_labels = [self._extract_mat_label(p) for p in roi_files]
			if self.var_brain.get():
				_ = self._extract_mat_label(self.var_brain.get(), is_brain=True)
			# GA mutation probability (0..1)
			try:
				ga_mut_prob = float(self.var_mut_prob.get())
				if ga_mut_prob < 0 or ga_mut_prob > 1:
					raise ValueError("Mutation probability must be in [0,1].")
			except Exception:
				ga_mut_prob = 0.1
			self.project = Project(
				name=self.var_name.get(),
				data_folder=self.var_folder.get(),
				brain_file=self.var_brain.get(),
				roi_files=roi_files,
				roi_labels=roi_labels,
				leadfield_files=lf_files,
				device_names=names,
				device_counts=counts,
				measure=self.var_measure.get(),
				method='genetic',
				montage=False,
				noise=noise,
				bandwidth=bw,
				scale=scale,
				angle_limit_rad=angle_rad,
				num_generations=int(self.var_gens.get()),
				sol_per_pop=int(self.var_init_solutions.get() or self.var_pop.get() or 100),
				num_parents_mating=int(self.var_parents.get()),
				process_count=int(self.var_proc.get()),
				depth_limits=depth_limits,
				proximity_limits=prox_limits,
				ga_mutation_prob=ga_mut_prob,
				device_lengths=device_lengths if 'device_lengths' in locals() else [],
				device_radii=device_radii if 'device_radii' in locals() else [],
				# Structural Avoidance persistence
				structural_files=list(getattr(self, 'lb_struct').get(0, 'end')) if hasattr(self, 'lb_struct') else [],
				structural_demo_enabled=bool(self.var_struct_demo.get()),
				structural_demo_radius_mm=float(self.var_struct_radius.get() or 5.0)
			)
			self.project.do_depth = self.var_do_depth.get()
			self.project.do_proximity = self.var_do_prox.get()
			self.lbl_status_setup.config(text=f"Project '{self.project.name}' ready")
			self._refresh_summary()
		except Exception as e:
			messagebox.showerror("Project Error", f"Failed to create/update project:\n{e}")
			traceback.print_exc()

	def _sync_project_from_form(self):
		"""Ensure self.project reflects all current GUI inputs before saving.

		This updates core metadata (name, folder, brain/ROI, leadfields) and
		per-device configuration (names, counts, noise, bandwidth, scale, angles,
		depth/proximity limits, physical dimensions) directly from the form.
		"""
		if not self.project:
			return
		try:
			p = self.project
			# Metadata
			p.name = (self.var_name.get() or '').strip() or p.name
			p.data_folder = (self.var_folder.get() or '').strip() or p.data_folder
			p.brain_file = (self.var_brain.get() or '').strip() or p.brain_file
			p.roi_files = list(self.lb_roi.get(0, 'end'))
			p.roi_labels = [self._extract_mat_label(path) for path in p.roi_files]
			p.leadfield_files = [self.tv_lf.item(it, 'values')[1] for it in self.tv_lf.get_children()]
			# Structural Avoidance
			p.structural_files = list(self.lb_struct.get(0, 'end')) if hasattr(self, 'lb_struct') else []
			p.structural_demo_enabled = bool(self.var_struct_demo.get())
			try:
				p.structural_demo_radius_mm = float(self.var_struct_radius.get() or 5.0)
			except Exception:
				p.structural_demo_radius_mm = 5.0
			# Device columns should match leadfield types
			lf_count = len(p.leadfield_files)
			if lf_count == 0:
				self.lbl_status_setup.config(text="No leadfields configured")
				return
			# If entry arrays out of sync, rebuild to avoid index errors
			if len(self.dev_name_entries) != lf_count:
				self._rebuild_device_grid()
			# Gather per-type fields
			names: list[str] = []
			counts: list[int] = []
			noise: list[float] = []
			bw: list[float] = []
			scale: list[float] = []
			angle_deg: list[float] = []
			depth_limits: list[tuple[float,float]] = []
			prox_limits: list[float] = []
			device_lengths: list[float] = []
			device_radii: list[float] = []
			for i in range(lf_count):
				# Safe getters
				def _f(ent, default):
					val = ent.get().strip() if ent else ''
					try:
						return float(val) if val != '' else default
					except Exception:
						return default
				def _i(ent, default):
					val = ent.get().strip() if ent else ''
					try:
						return int(val) if val != '' else default
					except Exception:
						return default
				name = (self.dev_name_entries[i].get().strip() if i < len(self.dev_name_entries) else '') or 'Device'
				names.append(name)
				counts.append(_i(self.dev_count_entries[i] if i < len(self.dev_count_entries) else None, 1))
				noise.append(_f(self.dev_noise_entries[i] if i < len(self.dev_noise_entries) else None, 2.3))
				bw.append(_f(self.dev_bw_entries[i] if i < len(self.dev_bw_entries) else None, 100.0))
				scale.append(_f(self.dev_scale_entries[i] if i < len(self.dev_scale_entries) else None, 0.5))
				angle_deg.append(_f(self.dev_angle_entries[i] if i < len(self.dev_angle_entries) else None, 0.0))
				# Depth limits
				dmin_raw = self.dev_depth_min_entries[i].get().strip() if i < len(self.dev_depth_min_entries) else ''
				dmax_raw = self.dev_depth_max_entries[i].get().strip() if i < len(self.dev_depth_max_entries) else ''
				mn = float(dmin_raw) if dmin_raw else np.nan
				mx = float(dmax_raw) if dmax_raw else np.nan
				depth_limits.append((mn, mx))
				# Proximity
				pmin_raw = self.dev_prox_entries[i].get().strip() if i < len(self.dev_prox_entries) else ''
				prox_limits.append(float(pmin_raw) if pmin_raw else (p.cl_offset if hasattr(p, 'cl_offset') else 3.0))
				# Physical dimensions
				len_raw = self.dev_length_entries[i].get().strip() if i < len(self.dev_length_entries) else ''
				rad_raw = self.dev_radius_entries[i].get().strip() if i < len(self.dev_radius_entries) else ''
				try:
					len_val = float(len_raw) if len_raw else 10.0
					if len_val <= 0: len_val = 10.0
				except Exception:
					len_val = 10.0
				try:
					rad_val = float(rad_raw) if rad_raw else 0.5
					if rad_val <= 0: rad_val = 0.5
				except Exception:
					rad_val = 0.5
				device_lengths.append(len_val)
				device_radii.append(rad_val)
			# Validation
			for idx, a in enumerate(angle_deg):
				if a < 0:
					raise ValueError(f"Angle limit cannot be negative (device type {idx+1}).")
			for idx, (mn, mx) in enumerate(depth_limits):
				if not np.isnan(mn) and not np.isnan(mx) and mn > mx:
					raise ValueError(f"Depth min exceeds max for device type {idx+1}.")
			for idx, pl in enumerate(prox_limits):
				if pl < 0:
					raise ValueError(f"Proximity limit cannot be negative (device type {idx+1}).")
			angle_rad = [a * np.pi / 180.0 for a in angle_deg]
			# Assign back to project
			p.device_names = names
			p.device_counts = counts
			p.noise = noise
			p.bandwidth = bw
			p.scale = scale
			p.angle_limit_rad = angle_rad
			p.depth_limits = depth_limits
			p.proximity_limits = prox_limits
			p.device_lengths = device_lengths
			p.device_radii = device_radii
			# Toggles
			p.do_depth = bool(self.var_do_depth.get())
			p.do_proximity = bool(self.var_do_prox.get())
		except Exception as e:
			messagebox.showerror("Form Sync Error", f"Invalid form values: {e}")
			raise

	def _save_project(self):
		if not self.project:
			messagebox.showwarning("No Project", "Create a project first.")
			return
		# First, sync core form inputs to the project so all user edits persist
		try:
			self._sync_project_from_form()
		except Exception:
			return
		# Then sync current optimization/toggle states before save
		try:
			# Common: initial solutions count used for seeding
			try:
				self.project.sol_per_pop = int(self.var_init_solutions.get())
			except Exception:
				pass
			self.project.do_depth = self.var_do_depth.get()
			self.project.do_proximity = self.var_do_prox.get()
			# Method and measure
			self.project.method = self.var_run_method.get()
			self.project.measure = self.var_measure.get()
			# Update physical dimensions from current grid entries
			lengths=[]; radii=[]
			for i in range(len(self.dev_length_entries)):
				try:
					lv=float(self.dev_length_entries[i].get().strip() or 10.0)
					if lv<=0: lv=10.0
				except Exception:
					lv=10.0
				try:
					rv=float(self.dev_radius_entries[i].get().strip() or 0.5)
					if rv<=0: rv=0.5
				except Exception:
					rv=0.5
				lengths.append(lv); radii.append(rv)
			setattr(self.project,'device_lengths', lengths)
			setattr(self.project,'device_radii', radii)
			# Parameters by method
			if self.project.method == 'anneal':
				self.project.anneal_iterations = int(self.var_a_iter.get())
				self.project.anneal_itemp = float(self.var_a_itemp.get())
				self.project.anneal_ftemp = float(self.var_a_ftemp.get())
				self.project.anneal_cooling_rate = float(self.var_a_cool.get())
				self.project.anneal_cart_step = float(self.var_a_cart.get())
				self.project.anneal_rot_step = float(self.var_a_rot.get())
			elif self.project.method == 'multianneal':
				self.project.anneal_iterations = int(self.var_a_iter.get())
				self.project.anneal_itemp = float(self.var_a_itemp.get())
				self.project.anneal_ftemp = float(self.var_a_ftemp.get())
				self.project.anneal_cooling_rate = float(self.var_a_cool.get())
				self.project.anneal_cart_step = float(self.var_a_cart.get())
				self.project.anneal_rot_step = float(self.var_a_rot.get())
				self.project.multi_anneal_restarts = int(self.var_ma_restarts.get())
			elif self.project.method == 'gradient':
				self.project.gradient_iterations = int(self.var_gd_iter.get())
				self.project.gradient_cart_step = float(self.var_gd_cart.get())
				self.project.gradient_rot_step = float(self.var_gd_rot.get())
				self.project.gradient_decay = float(self.var_gd_decay.get())
				self.project.gradient_simultaneous = int(self.var_gd_simult.get())
			elif self.project.method == 'branch_bound':
				self.project.branch_iterations = int(self.var_bb_iter.get())
				self.project.branch_instances = int(self.var_bb_instances.get())
				self.project.branch_top = int(self.var_bb_top.get())
				self.project.branch_angle_step = float(self.var_bb_angle.get())
				self.project.branch_cart_step = float(self.var_bb_cart.get())
				self.project.branch_threshold = float(self.var_bb_thresh.get())
				self.project.branch_decay = float(self.var_bb_decay.get())
			elif self.project.method == 'brute':
				self.project.brute_limit = int(self.var_bf_limit.get())
				self.project.brute_batch = int(self.var_bf_batch.get())
			else:
				self.project.num_generations = int(self.var_gens.get())
				self.project.sol_per_pop = int(self.var_init_solutions.get())
				self.project.num_parents_mating = int(self.var_parents.get())
				self.project.process_count = int(self.var_proc.get())
				mut = float(self.var_mut_prob.get())
				if mut < 0 or mut > 1:
					raise ValueError("Mutation probability must be in [0,1].")
				self.project.ga_mutation_prob = mut
		except Exception as e:
			messagebox.showerror("Save Error", f"Invalid optimization settings: {e}")
			return
		# Determine save path based on current metadata or loaded path
		try:
			# Case 1: Project was loaded, try to overwrite same file
			loaded_path = getattr(self.project, 'loaded_path', None)
			if loaded_path and os.path.isfile(loaded_path):
				self.project.save(loaded_path)
				self.lbl_status_setup.config(text=f"Saved: {os.path.basename(loaded_path)}")
				return
			# Case 2: Use Project Metadata (name + data folder)
			proj_name = (self.var_name.get() or '').strip()
			proj_folder = (self.var_folder.get() or '').strip()
			if proj_name and proj_folder:
				os.makedirs(proj_folder, exist_ok=True)
				path = os.path.join(proj_folder, f"{proj_name}.pkl")
				self.project.save(path)
				# track path for future overwrites
				self.project.loaded_path = path
				self.lbl_status_setup.config(text=f"Saved: {os.path.basename(path)}")
				return
			# Case 3: Fallback to prompt user
			f = filedialog.asksaveasfilename(initialfile=(proj_name or 'project.pkl'),
				defaultextension='.pkl', filetypes=[("Pickle","*.pkl")])
			if f:
				self.project.save(f)
				self.project.loaded_path = f
				self.lbl_status_setup.config(text=f"Saved: {os.path.basename(f)}")
			else:
				self.lbl_status_setup.config(text="Save canceled")
		except Exception as e:
			messagebox.showerror("Save Error", f"Could not save project:\n{e}")

	def _load_project(self):
		f = filedialog.askopenfilename(filetypes=[("Pickle","*.pkl")])
		if not f:
			return
		try:
			self.project = Project.load(f)
			# Track path for overwrite behavior
			setattr(self.project, 'loaded_path', f)
			# populate form
			p = self.project
			self.var_name.set(p.name)
			self.var_folder.set(p.data_folder)
			self.var_brain.set(p.brain_file)
			self.lb_roi.delete(0,'end')
			for rf in p.roi_files: self.lb_roi.insert('end', rf)
			# Structural Avoidance files
			if hasattr(self, 'lb_struct'):
				self.lb_struct.delete(0,'end')
				for sf in getattr(p, 'structural_files', []):
					self.lb_struct.insert('end', sf)
			# Clear any existing leadfields in the treeview
			for it in self.tv_lf.get_children():
				self.tv_lf.delete(it)
			for lf_idx, lf in enumerate(p.leadfield_files, start=1):
				self.tv_lf.insert('', 'end', values=(f"Type {lf_idx}", lf))
			# Prefill device grid from current-version project fields only
			self._rebuild_device_grid()
			self._prefill_device_grid(p)
			self.var_do_depth.set(getattr(p,'do_depth', True))
			self.var_do_prox.set(getattr(p,'do_proximity', True))
			# Structural Avoidance demo settings
			self.var_struct_demo.set(bool(getattr(p, 'structural_demo_enabled', False)))
			self.var_struct_radius.set(str(getattr(p, 'structural_demo_radius_mm', 5.0)))
			# Optimization method and parameters
			self.var_run_method.set(getattr(p, 'method', 'genetic'))
			self.var_measure.set(getattr(p, 'measure', 'ic'))
			# Rebuild param panel to reflect method selection
			self._rebuild_param_selection()
			# Genetic
			self.var_gens.set(str(getattr(p,'num_generations', 30)))
			self.var_pop.set(str(getattr(p,'sol_per_pop', 44)))
			# Common initial solutions mirrors sol_per_pop
			self.var_init_solutions.set(str(getattr(p,'sol_per_pop', 100)))
			self.var_parents.set(str(getattr(p,'num_parents_mating', 15)))
			self.var_proc.set(str(getattr(p,'process_count', 4)))
			self.var_mut_prob.set(str(getattr(p,'ga_mutation_prob', 0.1)))
			# Anneal
			self.var_a_iter.set(str(getattr(p,'anneal_iterations', 35)))
			self.var_a_itemp.set(str(getattr(p,'anneal_itemp', 100.0)))
			self.var_a_ftemp.set(str(getattr(p,'anneal_ftemp', 1e-3)))
			self.var_a_cool.set(str(getattr(p,'anneal_cooling_rate', 0.6)))
			self.var_a_cart.set(str(getattr(p,'anneal_cart_step', 15.0)))
			self.var_a_rot.set(str(getattr(p,'anneal_rot_step', np.pi/3)))
			# Multianneal
			self.var_ma_restarts.set(str(getattr(p,'multi_anneal_restarts', 5)))
			# Gradient
			self.var_gd_iter.set(str(getattr(p,'gradient_iterations', 200)))
			self.var_gd_cart.set(str(getattr(p,'gradient_cart_step', 10.0)))
			self.var_gd_rot.set(str(getattr(p,'gradient_rot_step', np.pi/3)))
			self.var_gd_decay.set(str(getattr(p,'gradient_decay', 0.99)))
			self.var_gd_simult.set(str(getattr(p,'gradient_simultaneous', -1)))
			# Branch-bound
			self.var_bb_iter.set(str(getattr(p,'branch_iterations', 36)))
			self.var_bb_instances.set(str(getattr(p,'branch_instances', 24)))
			self.var_bb_top.set(str(getattr(p,'branch_top', 6)))
			self.var_bb_angle.set(str(getattr(p,'branch_angle_step', np.pi/4)))
			self.var_bb_cart.set(str(getattr(p,'branch_cart_step', 8.0)))
			self.var_bb_thresh.set(str(getattr(p,'branch_threshold', 0.1)))
			self.var_bb_decay.set(str(getattr(p,'branch_decay', 0.95)))
			# Brute
			self.var_bf_limit.set(str(getattr(p,'brute_limit', 1000)))
			self.var_bf_batch.set(str(getattr(p,'brute_batch', 10)))
			self.lbl_status_setup.config(text=f"Loaded: {os.path.basename(f)}")
			self._refresh_summary()
			# Update Plotting tab selections and dropdowns from loaded project
			try:
				self._plot_refresh_listbox()
			except Exception:
				pass
		except Exception as e:
			messagebox.showerror("Load Error", f"Failed to load project:\n{e}")
			traceback.print_exc()

	def _prefill_device_grid(self, p: Project):
		lf_n = len(p.leadfield_files)
		# Ensure grid has lf_n columns
		self._rebuild_device_grid()
		for i in range(lf_n):
			if i < len(self.dev_name_entries): self.dev_name_entries[i].insert(0, p.device_names[i] if i < len(p.device_names) else '')
			if i < len(self.dev_count_entries): self.dev_count_entries[i].insert(0, str(p.device_counts[i] if i < len(p.device_counts) else 1))
			if i < len(self.dev_noise_entries): self.dev_noise_entries[i].insert(0, str(p.noise[i] if i < len(p.noise) else 2.3))
			if i < len(self.dev_bw_entries): self.dev_bw_entries[i].insert(0, str(p.bandwidth[i] if i < len(p.bandwidth) else 100.0))
			if i < len(self.dev_scale_entries): self.dev_scale_entries[i].insert(0, str(p.scale[i] if i < len(p.scale) else 0.5))
			if i < len(self.dev_angle_entries):
				deg = (p.angle_limit_rad[i]*180/np.pi) if i < len(p.angle_limit_rad) else 0.0
				self.dev_angle_entries[i].insert(0, str(round(deg,2)))
			if i < len(self.dev_depth_min_entries) and i < len(p.depth_limits):
				mn = p.depth_limits[i][0]
				self.dev_depth_min_entries[i].insert(0, '' if np.isnan(mn) else str(mn))
			if i < len(self.dev_depth_max_entries) and i < len(p.depth_limits):
				mx = p.depth_limits[i][1]
				self.dev_depth_max_entries[i].insert(0, '' if np.isnan(mx) else str(mx))
			if i < len(self.dev_prox_entries):
				val = p.proximity_limits[i] if i < len(p.proximity_limits) else p.cl_offset
				self.dev_prox_entries[i].insert(0, str(val))
			# Physical dimensions: length & radius
			if i < len(self.dev_length_entries):
				length_val = p.device_lengths[i] if hasattr(p, 'device_lengths') and i < len(getattr(p,'device_lengths',[])) else ''
				self.dev_length_entries[i].insert(0, str(length_val) if length_val != '' else '')
			if i < len(self.dev_radius_entries):
				radius_val = p.device_radii[i] if hasattr(p, 'device_radii') and i < len(getattr(p,'device_radii',[])) else ''
				self.dev_radius_entries[i].insert(0, str(radius_val) if radius_val != '' else '')

	# --------------------------------------------------------------
	# MAT label extraction (brain / ROI)
	# --------------------------------------------------------------
	def _extract_mat_label(self, path: str, is_brain: bool=False) -> str:
		"""Return the first non-meta variable name from a .mat file.

		If multiple arrays exist, warn the user and choose the first.
		If reading fails or no variables found, fallback to filename stem.
		"""
		if not os.path.isfile(path):
			return os.path.splitext(os.path.basename(path))[0]
		try:
			import scipy.io as sio  # local import to keep optional
			data = sio.loadmat(path)
			keys = [k for k in data.keys() if not k.startswith('__')]
			if not keys:
				return os.path.splitext(os.path.basename(path))[0]
			if len(keys) > 1:
				context = 'Brain' if is_brain else 'ROI'
				messagebox.showwarning(f"{context} MAT Variables", f"{os.path.basename(path)} contains multiple arrays: {keys}. Using first '{keys[0]}'.")
			return keys[0]
		except Exception as e:
			# Could be missing scipy or corrupted file
			context = 'Brain' if is_brain else 'ROI'
			messagebox.showwarning(f"{context} MAT Read", f"Failed reading {os.path.basename(path)}: {e}. Using filename stem.")
			return os.path.splitext(os.path.basename(path))[0]

	# --------------------------------------------------------------
	# Summary & result refresh
	# --------------------------------------------------------------
	def _refresh_summary(self):
		self.txt_summary.delete('1.0','end')
		if not self.project:
			self.txt_summary.insert('end', "No project configured.")
			return
		p = self.project
		lines = [
			f"Name: {p.name}",
			f"Brain: {p.brain_file}",
			f"ROIs ({len(p.roi_files)}):", *[f"  - {r}" for r in p.roi_files],
			f"Leadfields ({len(p.leadfield_files)}):", *[f"  - {l}" for l in p.leadfield_files],
			f"Devices: {p.device_counts} names={p.device_names}",
			f"Measure: {p.measure} Generations={p.num_generations} Pop={p.sol_per_pop}",
			f"Noise: {p.noise}",
			f"Bandwidth: {p.bandwidth}",
			f"Scale: {p.scale}",
			f"Angles(rad): {p.angle_limit_rad}",
			f"Depth Limits: {getattr(p,'depth_limits', [])}",
			f"Proximity Limits: {getattr(p,'proximity_limits', [])}",
			f"Device Lengths (mm): {getattr(p,'device_lengths', [])}",
			f"Device Radii (mm): {getattr(p,'device_radii', [])}",
			f"Depth Enabled: {getattr(p,'do_depth', True)}",
			f"Proximity Enabled: {getattr(p,'do_proximity', True)}",
		]
		if p.results:
			lines.append(f"Result: method={p.results.method} best_fitness={p.results.best_fitness:.2f}")
		self.txt_summary.insert('end', '\n'.join(lines))

	def _refresh_result(self):
		self.txt_result.delete('1.0','end')
		if not self.project or not self.project.results:
			self.txt_result.insert('end', "No optimization results available.")
			return
		r = self.project.results
		arr = r.best_solution.reshape((-1,6))
		lines = [f"Method: {r.method}", f"Best Fitness: {r.best_fitness:.2f}", f"Devices (x,y,z,alpha,beta,gamma):"]
		for i,row in enumerate(arr):
			lines.append(f"  {i+1}: " + ', '.join(f"{v:.3f}" for v in row))
		lines.append(f"Generations: {len(r.fitness_history)}")
		self.txt_result.insert('end', '\n'.join(lines))

	# --------------------------------------------------------------
	# Helpers
	# --------------------------------------------------------------
	def _matplotlib_required(self) -> bool:
		if not MATPLOTLIB_AVAILABLE:
			messagebox.showerror("Matplotlib", "matplotlib not installed.")
			return False
		return True

	def _plot_to_temp_png(self, fig, dpi: int = 120) -> str:
		import tempfile
		tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
		fig.savefig(tmp.name, dpi=dpi)
		if plt is not None:
			plt.close(fig)
		return tmp.name

	def _show_image_window(self, title: str, image_path: str, save_label: Optional[str] = None, save_cb: Optional[callable] = None):
		win = tk.Toplevel(self)
		win.title(title)
		img = tk.PhotoImage(file=image_path)
		lbl = ttk.Label(win, image=img)
		setattr(lbl, 'image', img)  # keep ref to avoid GC
		lbl.pack(padx=8, pady=8)
		if save_label and save_cb:
			ttk.Button(win, text=save_label, command=save_cb).pack(pady=6)
		return win

	def _ensure_project_and_folder(self) -> bool:
		if not getattr(self, 'project', None):
			messagebox.showwarning("No Project", "Create project first.")
			return False
		if not os.path.isdir(self.project.data_folder):
			messagebox.showerror("Folder", "Project data folder invalid.")
			return False
		return True

	def _copy_to_project(self, src_path: str, prefix: str, ext: str = ".png") -> str:
		stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
		dest = os.path.join(self.project.data_folder, f"{prefix}_{stamp}{ext}")
		with open(src_path, 'rb') as fsrc, open(dest, 'wb') as fdst:
			fdst.write(fsrc.read())
		return dest

	def _record_artifact(self, kind: str, path: str):
		proj = getattr(self, 'project', None)
		if proj is None:
			return
		if hasattr(proj, 'add_assessment_artifact'):
			try:
				proj.add_assessment_artifact(kind, path)
			except Exception:
				pass

	# --------------------------------------------------------------
	# Optimization run (threaded)
	# --------------------------------------------------------------
	def _start_optimization(self):
		if not self.project:
			messagebox.showwarning("No Project", "Configure project first.")
			return
		# Assign method selection
		# Sync constraint toggles to project in case user changed without recreating project
		self.project.do_depth = self.var_do_depth.get()
		self.project.do_proximity = self.var_do_prox.get()
		self.project.method = self.var_run_method.get()
		# Ensure measure is taken from Optimization tab
		self.project.measure = self.var_measure.get()
		# Common: initial solutions count
		try:
			self.project.sol_per_pop = int(self.var_init_solutions.get())
		except Exception:
			pass
		if self.project.method == 'anneal':
			try:
				self.project.anneal_iterations = int(self.var_a_iter.get())
				self.project.anneal_itemp = float(self.var_a_itemp.get())
				self.project.anneal_ftemp = float(self.var_a_ftemp.get())
				self.project.anneal_cooling_rate = float(self.var_a_cool.get())
				self.project.anneal_cart_step = float(self.var_a_cart.get())
				self.project.anneal_rot_step = float(self.var_a_rot.get())
			except ValueError:
				messagebox.showerror("Anneal Params", "Invalid annealing parameter values.")
				return
		elif self.project.method == 'gradient':
			try:
				self.project.gradient_iterations = int(self.var_gd_iter.get())
				self.project.gradient_cart_step = float(self.var_gd_cart.get())
				self.project.gradient_rot_step = float(self.var_gd_rot.get())
				self.project.gradient_decay = float(self.var_gd_decay.get())
				self.project.gradient_simultaneous = int(self.var_gd_simult.get())
			except ValueError:
				messagebox.showerror("Gradient Params", "Invalid gradient parameter values.")
				return
		elif self.project.method == 'mSGD':
			try:
				self.project.branch_iterations = int(self.var_bb_iter.get())
				self.project.branch_instances = int(self.var_bb_instances.get())
				self.project.branch_top = int(self.var_bb_top.get())
				self.project.branch_angle_step = float(self.var_bb_angle.get())
				self.project.branch_cart_step = float(self.var_bb_cart.get())
				self.project.branch_threshold = float(self.var_bb_thresh.get())
				self.project.branch_decay = float(self.var_bb_decay.get())
			except ValueError:
				messagebox.showerror("Branch-Bound Params", "Invalid branch-bound parameter values.")
				return
		elif self.project.method == 'brute':
			try:
				self.project.brute_limit = int(self.var_bf_limit.get())
				self.project.brute_batch = int(self.var_bf_batch.get())
			except ValueError:
				messagebox.showerror("Brute Params", "Invalid brute force parameter values.")
				return
		else:
			# Sync GA parameters from Optimization tab
			try:
				self.project.num_generations = int(self.var_gens.get())
				self.project.sol_per_pop = int(self.var_init_solutions.get())
				self.project.num_parents_mating = int(self.var_parents.get())
				self.project.process_count = int(self.var_proc.get())
				mut = float(self.var_mut_prob.get())
				if mut < 0 or mut > 1:
					raise ValueError
				self.project.ga_mutation_prob = mut
			except ValueError:
				messagebox.showerror("Genetic Params", "Invalid GA parameters (check numbers and mutation prob [0-1]).")
				return
		self.btn_run.config(state='disabled')
		self.lbl_opt_status.config(text="Running...")
		thread = threading.Thread(target=self._run_opt_thread, daemon=True)
		thread.start()

	def _run_opt_thread(self):
		try:
			start = datetime.datetime.now()
			# Set progress bar maximum using method-specific totals
			m = self.project.method
			if m == 'genetic':
				self.progress['maximum'] = max(1, self.project.num_generations)
			elif m == 'anneal':
				if self.project.anneal_cooling_rate <= 0 or self.project.anneal_cooling_rate >= 1:
					cool_steps = 1
				else:
					cool_steps = int(np.ceil(np.log(self.project.anneal_ftemp / self.project.anneal_itemp) / np.log(self.project.anneal_cooling_rate)))
					cool_steps = max(1, cool_steps)
				self.progress['maximum'] = cool_steps * self.project.anneal_iterations
			elif m == 'gradient':
				self.progress['maximum'] = max(1, getattr(self.project, 'gradient_iterations', 0))
			elif m == 'mSGD':
				self.progress['maximum'] = max(1, getattr(self.project, 'branch_iterations', 0))
			elif m == 'brute':
				# total = min(limit, sol_per_pop)
				limit = max(1, int(getattr(self.project, 'brute_limit', 0) or 0))
				self.progress['maximum'] = max(1, min(limit, int(self.project.sol_per_pop)))
			else:
				self.progress['maximum'] = 100

			def progress_cb(epoch: int, best: float, total: int):
				self.after(0, lambda e=epoch, b=best, t=total: self._update_progress(e,b,t))

			# Direct call with callback (real-time updates handled inside project methods)
			self.project.run_optimization(progress_callback=progress_cb)
			# Ensure stable labels are assigned before any save
			try:
				self._ensure_result_labels_persisted()
			except Exception:
				pass
			end = datetime.datetime.now()
			self.lbl_opt_status.config(text=f"Done ({(end-start).seconds}s)")
			# Auto-save project upon completion
			try:
				self._save_project()
			except Exception:
				pass
		except Exception as e:
			self.lbl_opt_status.config(text="Error")
			messagebox.showerror("Optimization Error", f"{e}\n\nTraceback:\n{traceback.format_exc()}")
		finally:
			self.btn_run.config(state='normal')
			self._refresh_summary()
			self._refresh_result_list()
			# Also refresh deletion dropdown on completion
			try:
				self._refresh_delete_list()
			except Exception:
				pass
			self._refresh_result()
			self.after(0, lambda: self.progress.stop())

	def _ensure_result_labels_persisted(self):
		proj = getattr(self, 'project', None)
		if not proj:
			return
		try:
			for r in getattr(proj, 'results_history', []) or []:
				lab = None
				try:
					lab = r.meta.get('label') if hasattr(r, 'meta') else None
				except Exception:
					lab = None
				if not isinstance(lab, str) or not lab.strip():
					# Calling this will compute and persist a stable unique label
					_ = self._label_for_result(r)
		except Exception:
			pass

	def _update_progress(self, epoch: int, best: float, total: int):
		# Adjust maximum if backend reports a different total
		try:
			if int(self.progress['maximum']) != int(total):
				self.progress['maximum'] = int(total)
		except Exception:
			pass
		self.progress['value'] = epoch
		self.lbl_opt_status.config(text=f"Epoch {epoch}/{total} Best {best:.2f}")

	def _label_for_result(self, res: OptimizationResult) -> str:
		# If a stable label exists in metadata, use it and do not change
		try:
			label = res.meta.get('label')
			if isinstance(label, str) and label.strip():
				return label
		except Exception:
			pass
		# Else compute a base label from method and key parameters
		try:
			stamp = res.meta.get('timestamp','')
			date = stamp.replace('-','').replace(':','').replace('T','')[:8]
		except Exception:
			date = ''
		if res.method == 'genetic':
			gens = res.meta.get('num_generations')
			pop = res.meta.get('sol_per_pop')
			mat = res.meta.get('num_parents_mating')
			proc = res.meta.get('process_count')
			base = f"Genetic-{date}-Gen{gens}-Pop{pop}-Mat{mat}-Proc{proc}"
		elif res.method == 'anneal':
			iters = res.meta.get('iterations', self.var_a_iter.get())
			cool = res.meta.get('cooling_rate')
			base = f"Anneal-{date}-Iter{iters}-Cool{cool}"
		elif res.method == 'gradient':
			iters = res.meta.get('iterations', self.var_gd_iter.get())
			base = f"Gradient-{date}-Iter{iters}"
		elif res.method == 'mSGD':
			iters = res.meta.get('iterations', self.var_bb_iter.get())
			inst = res.meta.get('instances', self.var_bb_instances.get() if hasattr(self, 'var_bb_instances') else '')
			base = f"mSGD-{date}-Iter{iters}-Inst{inst}"
		elif res.method == 'brute':
			evald = res.meta.get('evaluated', '')
			base = f"Brute-{date}-Eval{evald}"
		else:
			base = f"{res.method.capitalize()}-{date}"
		# Generate a stable, unique label and persist it into res.meta
		proj = getattr(self, 'project', None)
		labels_in_use = set()
		# In-memory labels
		try:
			for r in getattr(proj, 'results_history', []) or []:
				lab = r.meta.get('label') if hasattr(r, 'meta') else None
				if isinstance(lab, str) and lab:
					labels_in_use.add(lab)
		except Exception:
			pass
		# On-disk labels
		loaded_path = getattr(proj, 'loaded_path', None)
		if loaded_path and os.path.exists(loaded_path):
			try:
				proj_disk = Project.load(loaded_path)
				for r in getattr(proj_disk, 'results_history', []) or []:
					lab = r.meta.get('label') if hasattr(r, 'meta') else None
					if isinstance(lab, str) and lab:
						labels_in_use.add(lab)
			except Exception:
				pass
		# If base unused, take it; else find next available rerunX index
		candidate = base
		if candidate in labels_in_use:
			idx = 1
			while True:
				candidate = f"{base}-rerun{idx}"
				if candidate not in labels_in_use:
					break
				idx += 1
		# Persist stable label
		try:
			res.meta['label'] = candidate
		except Exception:
			pass
		return candidate

	def _label_base_only(self, res: OptimizationResult) -> str:
		try:
			stamp = res.meta.get('timestamp','')
			date = stamp.replace('-','').replace(':','').replace('T','')[:8]
		except Exception:
			date = ''
		if res.method == 'genetic':
			gens = res.meta.get('num_generations')
			pop = res.meta.get('sol_per_pop')
			mat = res.meta.get('num_parents_mating')
			proc = res.meta.get('process_count')
			return f"Genetic-{date}-Gen{gens}-Pop{pop}-Mat{mat}-Proc{proc}"
		elif res.method == 'anneal':
			iters = res.meta.get('iterations', self.var_a_iter.get())
			cool = res.meta.get('cooling_rate')
			return f"Anneal-{date}-Iter{iters}-Cool{cool}"
		elif res.method == 'gradient':
			iters = res.meta.get('iterations', self.var_gd_iter.get())
			return f"Gradient-{date}-Iter{iters}"
		elif res.method == 'mSGD':
			iters = res.meta.get('iterations', self.var_bb_iter.get())
			inst = res.meta.get('instances', self.var_bb_instances.get() if hasattr(self, 'var_bb_instances') else '')
			# Ensure method name consistency with mSGD labeling
			return f"mSGD-{date}-Iter{iters}-Inst{inst}"
		elif res.method == 'brute':
			evald = res.meta.get('evaluated', '')
			return f"Brute-{date}-Eval{evald}"
		else:
			return f"{res.method.capitalize()}-{date}"

	def _refresh_result_list(self):
		# Populate dropdown with all finished optimizations in project history
		items = []
		self._result_objects = []
		proj = getattr(self, 'project', None)
		if proj and hasattr(proj, 'results_history') and proj.results_history:
			for res in proj.results_history:
				items.append(self._label_for_result(res))
				self._result_objects.append(res)
		self.cmb_result_sel['values'] = items
		# Auto-select latest
		if items:
			self.var_result_sel.set(items[-1])
		else:
			self.var_result_sel.set('')

	def _refresh_delete_list(self):
		items = []
		self._del_result_objects = []
		proj = getattr(self, 'project', None)
		if proj and hasattr(proj, 'results_history') and proj.results_history:
			for res in proj.results_history:
				items.append(self._label_for_result(res))
				self._del_result_objects.append(res)
		self.cmb_del_result['values'] = items
		if items:
			self.var_del_result.set(items[-1])
		else:
			self.var_del_result.set('')

	def _delete_selected_result(self):
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		label = self.var_del_result.get()
		if not label:
			messagebox.showinfo("Delete", "No optimization selected.")
			return
		# Find selected result object
		selected = None
		for i, lab in enumerate(self.cmb_del_result['values']):
			if lab == label:
				try:
					selected = self._del_result_objects[i]
				except Exception:
					selected = None
				break
		if selected is None:
			messagebox.showerror("Delete", "Could not resolve selected optimization.")
			return
		# Confirm deletion with selected label for clarity
		confirm = messagebox.askyesno(
			"Confirm Deletion",
			f"Delete '{label}' permanently?"
		)
		if not confirm:
			return
		# Remove from history
		try:
			proj.results_history = [r for r in proj.results_history if r is not selected]
			if getattr(proj, 'results', None) is selected:
				proj.results = None
			# Save project and refresh lists
			try:
				self._save_project()
			except Exception:
				# Fallback to direct save if a validation error occurs during UI-driven save
				try:
					loaded_path = getattr(proj, 'loaded_path', None)
					if loaded_path:
						proj.save(loaded_path)
				except Exception:
					pass
			self._refresh_delete_list()
			self._refresh_result_list()
			self._refresh_result()
		except Exception as e:
			messagebox.showerror("Delete Error", f"Failed to remove result: {e}")

	def _clear_all_results(self):
		proj = getattr(self, 'project', None)
		if proj is None:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		# Check if there are any results to clear
		if not getattr(proj, 'results_history', None) and not getattr(proj, 'results', None):
			messagebox.showinfo("Clear All", "No optimization results to clear.")
			return
		# First confirmation
		ok1 = messagebox.askyesno(
			"Clear All Optimization Results",
			"This will delete ALL saved optimization results for this project. Continue?"
		)
		if not ok1:
			return
		# Second confirmation
		ok2 = messagebox.askyesno(
			"Confirm Permanent Deletion",
			"Are you absolutely sure? This action cannot be undone."
		)
		if not ok2:
			return
		try:
			# Clear all optimization results from the project
			proj.results_history = []
			proj.results = None
			# Persist changes
			try:
				self._save_project()
			except Exception:
				try:
					loaded_path = getattr(proj, 'loaded_path', None)
					if loaded_path:
						proj.save(loaded_path)
				except Exception:
					pass
			# Refresh UI elements
			self._refresh_delete_list()
			self._refresh_result_list()
			self._refresh_result()
			messagebox.showinfo("Clear All", "All optimization results have been cleared.")
		except Exception as e:
			messagebox.showerror("Clear All Error", f"Failed to clear all results: {e}")

	def _on_select_result(self):
		label = self.var_result_sel.get()
		if not label:
			return
		# Map selection to the corresponding result object
		for i, lab in enumerate(self.cmb_result_sel['values']):
			if lab == label:
				# Set selected result as current for display
				self._set_current_result(self._result_objects[i])
				break
		self._refresh_result()

	def _set_current_result(self, res: OptimizationResult):
		proj = getattr(self, 'project', None)
		if proj is None:
			return
		proj.results = res

	def _import_result_pickle(self):
		if not self.project:
			messagebox.showwarning("No Project", "Create or load a project first.")
			return
		# Prompt for a pickle file path since the UI field was removed
		path = filedialog.askopenfilename(filetypes=[("Pickle","*.pkl")])
		if not path:
			return
		if not os.path.isfile(path):
			messagebox.showerror("File", "Pickle file path invalid.")
			return
		try:
			# Delegate to model helper to support legacy formats
			if hasattr(self.project, 'import_result_pickle'):
				self.project.import_result_pickle(path)
				self.lbl_opt_status.config(text="Imported result")
				self._refresh_summary(); self._refresh_result()
			else:
				messagebox.showerror("Import", "Project does not support result import in this environment.")
		except Exception as e:
			messagebox.showerror("Import Error", f"Failed to import result:\n{e}")

	# --------------------------------------------------------------
	# Assessment actions
	# --------------------------------------------------------------
	def _show_fitness_plot(self):
		if not self._matplotlib_required():
			return
		if not self.project or not self.project.results or not self.project.results.fitness_history:
			messagebox.showwarning("No Data", "No fitness history to plot.")
			return
		data = self.project.results.fitness_history
		fig, ax = plt.subplots(figsize=(6,3))
		ax.plot(data, lw=2)
		ax.set_xlabel('Generation')
		ax.set_ylabel(self.project.measure.upper())
		ax.set_title('Fitness Over Generations')
		fig.tight_layout()
		img_path = self._plot_to_temp_png(fig, dpi=120)
		self._show_image_window("Fitness Plot", img_path, "Save to Project", lambda: self._save_plot_to_project(img_path))
		self.lbl_assess_status.config(text="Plot shown")

	def _save_plot_to_project(self, src_path: str):
		if not self._ensure_project_and_folder():
			return
		try:
			dest = self._copy_to_project(src_path, "fitness_plot", ".png")
			self._record_artifact('fitness_plot', dest)
			self.lbl_assess_status.config(text=f"Saved plot: {os.path.basename(dest)}")
		except Exception as e:
			messagebox.showerror("Save Error", f"Failed to save plot: {e}")

	def _export_best_csv(self):
		if not self.project or not self.project.results:
			messagebox.showwarning("No Result", "Run or load results first.")
			return
		out = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV","*.csv")])
		if not out:
			return
		try:
			export_best_solution_csv(self.project, out)
			if hasattr(self.project, 'add_assessment_artifact'):
				self.project.add_assessment_artifact('best_solution_csv', out)
			self.lbl_assess_status.config(text=f"Exported CSV: {os.path.basename(out)}")
		except Exception as e:
			messagebox.showerror("Export Error", f"Failed to export: {e}")

	def _export_epoch_csv(self):
		if not self.project or not self.project.results:
			messagebox.showwarning("No Result", "Run or load results first.")
			return
		out = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV","*.csv")])
		if not out:
			return
		try:
			if hasattr(self.project, 'export_epoch_history_csv'):
				self.project.export_epoch_history_csv(out)
				if hasattr(self.project, 'add_assessment_artifact'):
					self.project.add_assessment_artifact('epoch_history_csv', out)
				self.lbl_assess_status.config(text=f"Exported CSV: {os.path.basename(out)}")
			else:
				messagebox.showerror("Export Error", "Project does not support epoch history export in this environment.")
		except Exception as e:
			messagebox.showerror("Export Error", f"Failed to export: {e}")

	def _visualize_best_open3d(self):
		"""Open an Open3D window visualizing brain IC using best trajectories.

		This adapts methods from MDPO_visualize(NO SAVE).py via mdpo_project utilities.
		"""
		# Dependencies
		try:
			import open3d as o3d
		except Exception:
			messagebox.showerror("Open3D Missing", "Please install open3d to visualize: pip install open3d")
			return
		# Project/result guards
		p = self.project
		if not p or not getattr(p, 'results', None):
			messagebox.showwarning("No Result", "Run or load an optimization result first.")
			return
		best = p.results.best_solution
		if best is None or best.size == 0:
			messagebox.showwarning("No Trajectories", "Best solution is empty.")
			return
		# Import computational helpers from mdpo_project (geom only). We'll override TF/voltage to match MDPO_visualize_NO.py
		try:
			from GUI.mdpo_project import (
				vertex_normals, get_rotmat, normalize_faces, inflate_surface,
			)
		except Exception:
			messagebox.showerror("Module Import", "Failed to import mdpo_project helpers.")
			return

		# Mesh selection based on toggle
		use_roi = bool(getattr(self, 'var_only_roi', tk.IntVar(value=0)).get())
		if use_roi and (getattr(p, 'roi_vertices_full', None) is not None) and (getattr(p, 'roi_faces', None) is not None):
			vertices = np.asarray(p.roi_vertices_full)
			faces = np.asarray(p.roi_faces).astype(np.int64)
		else:
			if p.brain_vertices is None or p.brain_faces is None:
				messagebox.showwarning("Missing Mesh", "Brain mesh not cached. Please run optimization to cache geometry.")
				return
			vertices = np.asarray(p.brain_vertices)
			faces = np.asarray(p.brain_faces).astype(np.int64)
		# Ensure faces are zero-based and valid before computing normals
		faces = normalize_faces(faces, vertices.shape[0])
		normals = vertex_normals(vertices, faces)

		# Recenter vertices to brain-centered space like optimization
		try:
			brain_ref = np.asarray(p.brain_vertices) if getattr(p, 'brain_vertices', None) is not None else vertices
			center = np.mean(brain_ref, axis=0)
			recentered = vertices - center
		except Exception:
			print("Warning: Failed to recenter mesh; using original coordinates.")
			recentered = vertices

		# Compute dipole layer by offset along normals using project parameter
		dist = float(getattr(p, 'dipole_offset', 0.5))
		layer4 = inflate_surface(recentered, normals, dist)

		# Leadfield: load first file via FieldImporter
		try:
			from modules.leadfield_importer import FieldImporter
			fi = FieldImporter()
			if not p.leadfield_files:
				messagebox.showwarning("Leadfield", "No leadfield files in project.")
				return
			fi.load(p.leadfield_files[0])
			fields = fi.fields
		except Exception as e:
			messagebox.showerror("Leadfield Load", f"Failed to load leadfield: {e}")
			return

		# Parameters
		scale = float(p.scale[0]) if p.scale else 0.5
		magnitude = float(getattr(p, 'magnitude', 0.5e-9))
		noise = float(p.noise[0]) if p.noise else 2.7
		bandwidth = float(p.bandwidth[0]) if p.bandwidth else 100.0
		montage = bool(getattr(p, 'montage', False))

		# Trajectories
		devpos = best.reshape((-1,6))

		# Compute IC per vertex using the same mdpo_project helpers as optimization
		try:
			ic_max = np.full((layer4.shape[0],), -np.inf, dtype=float)
			for dev in devpos:
				pos0, vec0 = transform_vectorspace(fields, scale, magnitude, layer4, normals, dev)
				pos1, vec1 = trim_data(fields, pos0, vec0)
				_, _, ic_vals = calculate_voltage(fields, pos1, vec1, v_scale=1e6, noise=noise, bandwidth=bandwidth, weights=None, montage=montage)
				ic_max = np.maximum(ic_max, np.nan_to_num(ic_vals, nan=-np.inf))
			ic_max = np.where(np.isfinite(ic_max), ic_max, 0.0)
		except Exception as e:
			messagebox.showerror("Compute Error", f"Failed IC computation: {e}")
			return

		# Build IC bins (7 bins) and map to colors using standalone-style palette
		vals = np.nan_to_num(ic_max)
		mn, mx = float(np.min(vals)), float(np.max(vals))
		if not np.isfinite(mx) or mx <= 0:
			mx = 1e-9
		# 7 bins with geometric boundaries: each boundary is 1/5 of the previous (descending), lowest at 0
		# Build ascending edges: [0, mx/5^6, mx/5^5, ..., mx/5, mx]
		ratio = 5.0
		geo_edges = [0.0] + [mx / (ratio ** k) for k in range(6, 0, -1)] + [mx]
		edges = np.array(geo_edges, dtype=float)
		# Palette derived from mdpo_project get_palette (anchors), interpolated to 7 colors
		try:
			from GUI.mdpo_project import get_palette
			from matplotlib.colors import LinearSegmentedColormap, to_rgb
			anchors = get_palette('Contrast')[1:]  # drop base gray
			cmap_bins = LinearSegmentedColormap.from_list('mdpo_contrast', [to_rgb(h) for h in anchors], N=7)
			bin_colors = np.asarray(cmap_bins(np.linspace(0,1,7))[:, :3], dtype=float)
		except Exception:
			# Fallback simple gradient (7 colors)
			bin_colors = np.stack([
				np.linspace(0.2, 0.9, 7),
				np.linspace(0.2, 0.6, 7),
				np.linspace(0.9, 0.2, 7)
			], axis=1).astype(float)
		# Digitize values into bins 0..6
		idx = np.digitize(vals, edges[1:], right=True)
		idx = np.clip(idx, 0, 6)
		# Reverse color order per request (highest IC gets first color in reversed list)
		bin_colors = bin_colors[::-1]
		vert_colors = bin_colors[idx]
		# If colors degenerate (e.g., all same), fallback to continuous mapping
		if np.allclose(vert_colors.max(axis=0), vert_colors.min(axis=0)):
			norm = (vals - mn) / (mx - mn if mx > mn else 1.0)
			vert_colors = np.stack([norm, 1.0 - norm, 0.5*np.ones_like(norm)], axis=1)
		# Build Open3D mesh and color per-vertex
		mesh = o3d.geometry.TriangleMesh()
		# Visualize on recentered mesh geometry; colors reflect layer4 IC values
		mesh.vertices = o3d.utility.Vector3dVector(recentered)
		mesh.triangles = o3d.utility.Vector3iVector(faces)
		mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
		# Ensure normals point outward and triangle winding is consistent for opaque rendering
		try:
			mesh.compute_triangle_normals()
			# Some Open3D versions have orient_triangles to fix winding
			if hasattr(mesh, 'orient_triangles'):
				mesh.orient_triangles()
			mesh.compute_vertex_normals()
		except Exception:
			mesh.compute_vertex_normals()

		geoms = [mesh]

		# Add structural avoidance spheres (wireframe), gated by Show Avoidance
		try:
			show_avoid = bool(getattr(self, 'var_show_avoidance', tk.IntVar(value=1)).get())
			if show_avoid:
				avoid = self.structural_avoidance if self.structural_avoidance is not None else self._compute_structural_avoidance()
				if avoid is not None and avoid.size > 0:
					for row in np.asarray(avoid):
						x, y, z, r = float(row[0]), float(row[1]), float(row[2]), float(row[3])
						if not np.isfinite(r) or r <= 0:
							continue
						sp = o3d.geometry.TriangleMesh.create_sphere(radius=r)
						sp.compute_vertex_normals()
						sp.translate(np.array([x, y, z], dtype=float))
						# Convert mesh triangles to line set for wireframe effect
						pts = np.asarray(sp.vertices)
						tris = np.asarray(sp.triangles, dtype=np.int64)
						edges = set()
						for a, b, c in tris:
							edges.add(tuple(sorted((int(a), int(b)))))
							edges.add(tuple(sorted((int(b), int(c)))))
							edges.add(tuple(sorted((int(c), int(a)))))
						lines = np.array(list(edges), dtype=np.int64)
						ls = o3d.geometry.LineSet()
						ls.points = o3d.utility.Vector3dVector(pts)
						ls.lines = o3d.utility.Vector2iVector(lines)
						ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.1, 0.7, 1.0]]), (lines.shape[0], 1)))
						geoms.append(ls)
		except Exception:
			pass

		# Optional debug markers: mesh centroid and device start points
		do_debug = bool(getattr(self, 'var_debug', tk.IntVar(value=0)).get())
		if do_debug:
			try:
				import open3d as o3d
				# Centroid marker (small sphere at origin of recentered space)
				centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
				centroid_sphere.paint_uniform_color([0.1, 0.8, 0.1])
				centroid_sphere.translate(np.array([0.0, 0.0, 0.0]))
				geoms.append(centroid_sphere)
			except Exception:
				pass

		# Add cylinders for devices
		try:
			for i in range(devpos.shape[0]):
				pos = devpos[i]
				radius = float(p.device_radii[i]) if p.device_radii and i < len(p.device_radii) else 0.5
				length = float(p.device_lengths[i]) if p.device_lengths and i < len(p.device_lengths) else 10.0
				alpha, beta, gamma = float(pos[3]), float(pos[4]), float(pos[5])
				R = get_rotmat(alpha, beta, gamma)
				# Device local +Z axis mapped to source via forward rotation
				axis = (R @ np.array([0.0, 0.0, 1.0]))
				axis = axis / (np.linalg.norm(axis) + 1e-12)
				# Device positions are already in the recentered brain-centered frame; no midpoint offsets
				start_point = np.array([pos[0], pos[1], pos[2]], dtype=float)
				# Front cylinder (yellow)
				cyl_f = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
				cyl_f.paint_uniform_color([1.0, 0.85, 0.0])
				cyl_f.rotate(R, center=np.array([0.0, 0.0, 0.0]))
				cyl_f.translate(start_point + axis * (length/2.0))
				geoms.append(cyl_f)
				# Back cylinder (orange), same length
				cyl_b = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
				cyl_b.paint_uniform_color([1.0, 0.55, 0.0])
				cyl_b.rotate(R, center=np.array([0.0, 0.0, 0.0]))
				cyl_b.translate(start_point - axis * (length/2.0))
				geoms.append(cyl_b)

				# Debug: visualize device local axes (X,Y,Z) transformed by R
				if do_debug:
					try:
						# Small arrows/cylinders to show orientation
						axis_len = max(5.0, length * 0.2)
						# X-axis (red)
						x_dir = (R @ np.array([1.0, 0.0, 0.0]))
						x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
						cyl_x = o3d.geometry.TriangleMesh.create_cylinder(radius=radius*0.3, height=axis_len)
						cyl_x.paint_uniform_color([0.9, 0.1, 0.1])
						# Align cylinder along x_dir
						# Build rotation that maps +Z to x_dir: R_align = R @ R_z_to_x, but simpler: rotate identity cylinder by matrix that has columns as axes
						# We reuse R and place along start + x_dir * axis_len/2
						# Create rotation matrix that maps local Z to x_dir
						# Use orthonormal basis from R columns
						R_basis = R
						# For X, swap basis so Z aligns to X
						R_x = np.array([R_basis[:,2], R_basis[:,1], R_basis[:,0]]).T
						cyl_x.rotate(R_x, center=np.array([0.0, 0.0, 0.0]))
						cyl_x.translate(start_point + x_dir * (axis_len/2.0))
						geoms.append(cyl_x)
						# Y-axis (green)
						y_dir = (R @ np.array([0.0, 1.0, 0.0]))
						y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)
						cyl_y = o3d.geometry.TriangleMesh.create_cylinder(radius=radius*0.3, height=axis_len)
						cyl_y.paint_uniform_color([0.1, 0.9, 0.1])
						R_y = np.array([R_basis[:,0], R_basis[:,2], R_basis[:,1]]).T
						cyl_y.rotate(R_y, center=np.array([0.0, 0.0, 0.0]))
						cyl_y.translate(start_point + y_dir * (axis_len/2.0))
						geoms.append(cyl_y)
						# Z-axis (blue)
						z_dir = axis
						cyl_z = o3d.geometry.TriangleMesh.create_cylinder(radius=radius*0.3, height=axis_len)
						cyl_z.paint_uniform_color([0.2, 0.2, 0.95])
						cyl_z.rotate(R, center=np.array([0.0, 0.0, 0.0]))
						cyl_z.translate(start_point + z_dir * (axis_len/2.0))
						geoms.append(cyl_z)
						# Sphere at raw device positions
						origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.75, radius))
						origin_sphere.paint_uniform_color([0.9, 0.1, 0.1])
						origin_sphere.translate(np.array([pos[0], pos[1], pos[2]]))
						geoms.append(origin_sphere)
					except Exception:
						pass
					try:
						# Arrow from origin toward device position (first 3 elements)
						dir_vec = start_point.astype(float)
						dir_len = float(np.linalg.norm(dir_vec))
						if dir_len > 1e-9:
							dir_unit = dir_vec / dir_len
							arrow_len = dir_len
							arrow_rad = max(0.2, radius * 0.4)
							arr = o3d.geometry.TriangleMesh.create_arrow(
								cylinder_radius=arrow_rad,
								cone_radius=arrow_rad * 1.6,
								cylinder_height=arrow_len * 0.75,
								cone_height=arrow_len * 0.25
							)
							arr.paint_uniform_color([0.2, 0.8, 1.0])
							# Rotate local +Z to dir_unit using Rodrigues' rotation formula
							z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
							cross = np.cross(z_axis, dir_unit)
							s = np.linalg.norm(cross)
							if s < 1e-9:
								# Parallel or anti-parallel to +Z
								if np.dot(z_axis, dir_unit) >= 0:
									R_arrow = np.eye(3)
								else:
									# 180° around X axis maps +Z -> -Z
									R_arrow = np.array([[1.0, 0.0, 0.0],
														[0.0, -1.0, 0.0],
														[0.0, 0.0, -1.0]], dtype=float)
							else:
								k = cross / s
								c = float(np.dot(z_axis, dir_unit))
								K = np.array([[0, -k[2], k[1]],
											  [k[2], 0, -k[0]],
											  [-k[1], k[0], 0]], dtype=float)
								R_arrow = np.eye(3) + K * s + (K @ K) * (1.0 - c)
							arr.rotate(R_arrow, center=np.array([0.0, 0.0, 0.0]))
							# Tail at origin, head toward device position
							geoms.append(arr)
					except Exception:
						pass
				# Debug: small sphere at device start
				if do_debug:
					try:
						start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.75)
						start_sphere.paint_uniform_color([0.9, 0.1, 0.1])
						start_sphere.translate(start_point)
						geoms.append(start_sphere)
					except Exception:
						pass
		except Exception:
			pass

		# Show 3D viewer using legacy API to avoid GUI init issues
		try:
			import open3d as o3d
			# Flip logic per user request: checked means opaque (no back faces), unchecked means X-Ray
			xray_checked = bool(getattr(self, 'var_xray', tk.IntVar(value=0)).get())
			show_back_faces = (not xray_checked)
			# In X-Ray mode, show back faces so inner surfaces become visible; otherwise cull back faces
			o3d.visualization.draw_geometries(
				geoms,
				window_name="Brain Information Capacity",
				mesh_show_back_face=show_back_faces
			)
		except Exception:
			pass
		self._show_ic_legend_window(edges, bin_colors)
		self.lbl_assess_status.config(text="Open3D view shown")

	def _show_ic_legend_window(self, edges: np.ndarray, bin_colors: np.ndarray):
		"""Show IC color scale legend in a separate Matplotlib window."""
		try:
			import matplotlib.pyplot as plt
			from matplotlib.colors import ListedColormap, BoundaryNorm
			from matplotlib.cm import ScalarMappable
			# Build discrete colormap and boundary norm; ensure Python list input for colormap
			cmap = ListedColormap(np.asarray(bin_colors, dtype=float).tolist())
			norm = BoundaryNorm(edges, ncolors=len(bin_colors))
			fig, ax = plt.subplots(figsize=(8, 1.4))
			sm = ScalarMappable(cmap=cmap, norm=norm)
			sm.set_array([])
			cb = plt.colorbar(sm, cax=ax, orientation='horizontal', ticks=edges)
			# Label each bin boundary
			cb.set_ticks(edges)
			cb.set_ticklabels([f"{e:.3g}" for e in edges])
			cb.set_label('IC (bits/s)')
			fig.tight_layout()
			plt.show(block=False)
		except Exception:
			pass

	def _open3d_show_with_legend(self, geoms, edges: np.ndarray, bin_colors: np.ndarray, title: str = "Open3D", xray: bool = False):
		# Try import base Open3D first
		try:
			import open3d as o3d
		except Exception:
			return  # Cannot render at all
		# Legacy fallback path only; GUI renderer disabled due to initialization issues
		try:
			# Flip logic: xray flag means checkbox checked; show back faces when unchecked
			o3d.visualization.draw_geometries(geoms, window_name=title, mesh_show_back_face=(not xray))
		except Exception:
			pass
		return

		app = gui.Application.instance
		# Initialize once; ignore if already initialized
		try:
			app.initialize()
		except Exception:
			pass
		win = app.create_window(title, 1024, 768)
		scene = gui.SceneWidget()
		scene.scene = rendering.Open3DScene(win.renderer)
		# GUI rendering path removed; relying on legacy viewer only
		return
		bbox = None
		for g in geoms:
			gb = g.get_axis_aligned_bounding_box()
			bbox = gb if bbox is None else bbox + gb
		if bbox is not None:
			scene.setup_camera(60.0, bbox, bbox.get_center())
		win.add_child(scene)

		legend = gui.Vert(2, gui.Margins(6, 6, 6, 6))
		title_lbl = gui.Label("IC (bits/s)")
		legend.add_child(title_lbl)
		for i in range(8):
			row = gui.Horiz(4)
			c = gui.ColorEdit()
			c.set_value(gui.Color(float(bin_colors[i,0]), float(bin_colors[i,1]), float(bin_colors[i,2])))
			c.enabled = False
			row.add_child(c)
			row.add_child(gui.Label(f"{edges[i]:.2f} – {edges[i+1]:.2f}"))
			legend.add_child(row)

		def on_layout(ctx):
			pref = legend.calc_preferred_size(ctx, gui.Widget.Constraints())
			legend.frame = gui.Rect(win.content_rect.get_right() - pref.width - 10,
									win.content_rect.y + 10,
									pref.width, pref.height)
		win.set_on_layout(on_layout)
		win.add_child(legend)
		app.run()

	def _show_scatter_plot(self):
		if not self._matplotlib_required():
			return
		if not self.project or not self.project.results or self.project.roi_recentered is None:
			messagebox.showwarning("Missing Data", "Need optimization run to cache ROI geometry.")
			return
		pts = self.project.roi_recentered
		if pts.shape[0] > 5000:
			sel = np.random.choice(pts.shape[0], 5000, replace=False)
			pts_plot = pts[sel]
		else:
			pts_plot = pts
		devices = self.project.results.best_solution.reshape((-1,6))[:, :3]
		fig = plt.figure(figsize=(5,4))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(pts_plot[:,0], pts_plot[:,1], pts_plot[:,2], s=2, c='lightgray', alpha=0.5)
		ax.scatter(devices[:,0], devices[:,1], devices[:,2], s=80, c='red', depthshade=True)
		ax.set_title('Devices vs ROI')
		ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
		fig.tight_layout()
		img_path = self._plot_to_temp_png(fig, dpi=120)
		self._show_image_window("3D Scatter", img_path, "Save to Project", lambda: self._save_scatter(img_path))
		self.lbl_assess_status.config(text="Scatter shown")


	def _show_rainbow_legend(self):
		"""Popup legend indicating rainbow spectrum from Start to End."""
		win = tk.Toplevel(self)
		win.title("Trajectory Color Legend")
		frm = ttk.Frame(win)
		frm.pack(padx=10, pady=10)
		# Build gradient canvas using matplotlib if available
		width, height = 240, 24
		try:
			import matplotlib.cm as cm
			cmap = cm.get_cmap('rainbow')
			grad = np.linspace(0,1,width)
			rgb = (cmap(grad)[:,:3]*255).astype(np.uint8)
			raster = np.tile(rgb[np.newaxis, :, :], (height, 1, 1))
			from PIL import Image, ImageTk  # Pillow likely present in scientific envs
			img = Image.fromarray(raster)
			photo = ImageTk.PhotoImage(img)
			canvas = tk.Label(frm, image=photo)
			canvas.image = photo
			canvas.pack()
		except Exception:
			# Simple textual fallback
			ttk.Label(frm, text="Start  —  End", foreground='black').pack()
		# Labels
		row = ttk.Frame(frm); row.pack(fill='x', pady=6)
		ttk.Label(row, text="Start", foreground='black').pack(side='left')
		ttk.Label(row, text="End", foreground='black').pack(side='right')

	def _save_scatter(self, src_path: str):
		if not self._ensure_project_and_folder():
			return
		try:
			dest = self._copy_to_project(src_path, "scatter", ".png")
			self._record_artifact('scatter_plot', dest)
			self.lbl_assess_status.config(text="Scatter saved")
		except Exception as e:
			messagebox.showerror("Save Error", f"Failed saving scatter: {e}")


def main():
	app = MDPOGUI()
	app.mainloop()


if __name__ == '__main__':
	# Reciting the Litany of Awakening...
	main()

