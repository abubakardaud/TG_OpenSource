"""
Different Negative Edge Sampling Approaches for Dynamic Graphs

Date: Oct. 30th, 2022
"""

import numpy as np
import torch



class Rand_Negative_Edge_Sampler(object):
	def __init__(self, input_dict, seed=None):
		if seed is not None:
			self.seed = seed
			self.random_state = np.random.RandomState(self.seed)
		else:
			self.seed = None

		self.neg_sample = input_dict['neg_sample']
		if self.neg_sample not in ['rnd', 'hist', 'induc']:
			raise RuntimeError("Negative sampling Strategy should be one of 'rnd', 'hist, or 'induc'.")

		if self.neg_sample == 'rnd':
			self._initialize_rnd(input_dict)
		else:
			self.initialize_not_rnd(input_dict)

	def reset_random_state(self):
		self.random_state = np.random.RandomState(self.seed)

	def _initialize_rnd(self, input_dict):
		self.src_list = np.unique(input_dict['src_list'])
		self.dst_list = np.unique(input_dict['dst_list'])
	
	def _initialize_not_rnd(self, input_dict):
		self.rnd_sample_ratio = input_dict['rnd_sample_ratio']
		self.src_list = input_dict['src_list']
		self.dst_list = input_dict['dst_list']
		self.ts_list = input_dict['ts_list']
		self.src_list_distinct = np.unique(self.src_list)
		self.dst_list_distinct = np.unique(self.dst_list)
		self.ts_list_distinct = np.unique(self.ts_list)
		self.ts_init = min(self.ts_list_distinct)
		self.ts_end = max(self.ts_list_distinct)
		self.ts_test_split = input_dict['last_ts_train_val']
		self.e_train_val_l = self._get_edges_in_time_interval(self.ts_init, self.ts_test_split)


	def _get_edges_in_time_interval(self, start_ts, end_ts):
		"""
	    return edges of a specific time interval
	    """
		valid_ts_interval = (self.ts_list >= start_ts) * (self.ts_list <= end_ts)
		interval_src_l = self.src_list[valid_ts_interval]
		interval_dst_l = self.dst_list[valid_ts_interval]
		interval_edges = {}
		for src, dst in zip(interval_src_l, interval_dst_l):
			if (src, dst) not in interval_edges:
				interval_edges[(src, dst)] = 1
		return interval_edges


	def _get_difference_edge_list(self, first_e_set, second_e_set):
		"""
	    return edges in the first_e_set that are not in the second_e_set
	    """
		difference_e_set = set(first_e_set) - set(second_e_set)
		src_l, dst_l = [], []
		for e in difference_e_set:
			src_l.append(e[0])
			dst_l.append(e[1])
		return np.array(src_l), np.array(dst_l)

	
	def _rnd_sample(self, size):
		if self.seed is None:
			src_index = np.random.randint(0, len(self.src_list), size)
			dst_index = np.random.randint(0, len(self.dst_list), size)
		else:
			src_index = self.random_state.randint(0, len(self.src_list), size)
			dst_index = self.random_state.randint(0, len(self.dst_list), size)
		return self.src_list[src_index], self.dst_list[dst_index]
	
	def _hist_sample(self, size, current_split_start_ts, current_split_end_ts):
		history_e_dict = self._get_edges_in_time_interval(self.ts_init, current_split_start_ts)
		current_split_e_dict = self._get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
		non_repeating_e_src_l, non_repeating_e_dst_l = self._get_difference_edge_list(history_e_dict, current_split_e_dict)
		
		num_smp_rnd = int(self.rnd_sample_ratio * size)
		num_smp_from_hist = size - num_smp_rnd
		if num_smp_from_hist > len(non_repeating_e_src_l):
			num_smp_from_hist = len(non_repeating_e_src_l)
			num_smp_rnd = size - num_smp_from_hist
			
		replace = len(self.src_list_distinct) < num_smp_rnd
		rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
		
		replace = len(self.dst_list_distinct) < num_smp_rnd
		rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)
		
		replace = len(non_repeating_e_src_l) < num_smp_from_hist
		nre_e_index = np.random.choice(len(non_repeating_e_src_l), size=num_smp_from_hist, replace=replace)
		
		negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], non_repeating_e_src_l[nre_e_index]])
		negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], non_repeating_e_dst_l[nre_e_index]])
		return negative_src_l, negative_dst_l
		
	def _induc_sample(self, size, current_split_start_ts, current_split_end_ts):
		history_e_dict = self._get_edges_in_time_interval(self.ts_init, current_split_start_ts)
		current_split_e_dict = self._get_edges_in_time_interval(current_split_start_ts, current_split_end_ts)
		induc_non_repeat_e = set(set(history_e_dict) - set(self.e_train_val_l)) - set(current_split_e_dict)
		induc_adv_src_l, induc_adv_dst_l = [], []
		if len(induc_non_repeat_e) > 0:
			for e in induc_non_repeat_e:
				induc_adv_src_l.append(e[0])
				induc_adv_dst_l.append(e[1])
			induc_adv_src_l = np.array(induc_adv_src_l)
			induc_adv_dst_l = np.array(induc_adv_dst_l)
		
		num_smp_rnd = size - len(induc_non_repeat_e)
		if num_smp_rnd > 0:
			replace = len(self.src_list_distinct) < num_smp_rnd
			rnd_src_index = np.random.choice(len(self.src_list_distinct), size=num_smp_rnd, replace=replace)
			replace = len(self.dst_list_distinct) < num_smp_rnd
			rnd_dst_index = np.random.choice(len(self.dst_list_distinct), size=num_smp_rnd, replace=replace)
			
			negative_src_l = np.concatenate([self.src_list_distinct[rnd_src_index], induc_adv_src_l])
			negative_dst_l = np.concatenate([self.dst_list_distinct[rnd_dst_index], induc_adv_dst_l])
		else:
			rnd_induc_hist_index = np.random.choice(len(induc_non_repeat_e), size=size, replace=False)
			negative_src_l = induc_adv_src_l[rnd_induc_hist_index]
			negative_dst_l = induc_adv_dst_l[rnd_induc_hist_index]
		return negative_src_l, negative_dst_l
		
	def sample(self, sample_input):
		if self.neg_sample == 'rnd':
			neg_src_list, neg_dst_list = self._rnd_sample(sample_input['size'])
		elif self.neg_sample == 'hist':
			neg_src_list, neg_dst_list = self._hist_sample(sample_input['size'], sample_input['current_split_start_ts'], sample_input['current_split_end_ts'])
		elif self.neg_sample == 'induc':
			neg_src_list, neg_dst_list = self._induc_sample(sample_input['size'], sample_input['current_split_start_ts'], sample_input['current_split_end_ts'])
		else:
			raise ValueError("Undefined Negative Sampling Strategy!")
		return neg_src_list, neg_dst_list

	

	