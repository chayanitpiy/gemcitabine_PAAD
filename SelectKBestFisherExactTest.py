#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
Version: 1.0

Author: Alexandra Bomane <alexandra.bomane@inserm.fr>

This class is adapted from the class SelectKBest provided by Scikit-Learn:
https://scikit-learn.org/stable/modules/generated/
	sklearn.feature_selection.SelectKBest.html
Allows to make feature selection on p-values obtained
	from Fisher's Exact Test.
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import sys
from warnings import warn

########################################################################

class SelectKBestPvalueFisherExact:
	"""Select features according to the k lowest p-values obtained by
	Fisher's exact test.
	Parameters
	----------
	k : int or "all", optional, default = 10
		Number of top features to select.
		The "all" option bypasses selection, for use in a parameter search.
	Attributes
	----------
	scores_ : array-like, shape=(n_features,)
		Scores of features.
	pvalues_ : array-like, shape=(n_features,)
		p-values of feature scores, None if `score_func` returned only scores.
	"""

	def __init__(self, k = 10):
		self.k = k
		self.scores_ = np.zeros(self.k)
		self.pvalues_ = np.zeros(self.k)

	def calculate(self, X, y):
		# ValueError: cannot set WRITEABLE flag to True of this array
		#X.flags['WRITEABLE'] = True
		#y.flags['WRITEABLE'] = True

		Xc = X.copy()
		yc = y.copy()
		res = np.array([fisher_exact(pd.crosstab(pd.Categorical(yc, categories = [0,1]), pd.Categorical(Xc[:,j], categories = [0,1]), dropna = False)) for j in range(Xc.shape[1])])
		del Xc
		del yc

		#X.flags['WRITEABLE'] = False
		#y.flags['WRITEABLE'] = False

		self.scores_ = res[:,0]
		self.pvalues_ = res[:,1]

	def get_params(self, deep = True):
		if deep:
			return {'k':self.k}
		else:
			return {'k':self.k}

	def set_params(self, **params):
		if not params:
			# Simple optimization to gain speed (inspect is slow)
			return self

		valid_params = self.get_params(deep = True)

		if len(params) == 1:
			if 'k' not in params.keys():
				raise ValueError('Invalid parameter %s for estimator %s. '
				'Check the list of available parameters '
				'with `estimator.get_params().keys()`.' %(key, self))
			else:
				self.k = params['k']
		else:
			print("Too many params for SelectKBestPvalueFisherExact object")
			sys.exit(0)

	def fit(self, X, y):
		if not self.pvalues_.any():

			#X.flags['WRITEABLE'] = True
			#y.flags['WRITEABLE'] = True

			Xc = X.copy()
			yc = y.copy()
			res = np.array([fisher_exact(pd.crosstab(pd.Categorical(yc, categories = [0,1]), pd.Categorical(Xc[:,j], categories = [0,1]), dropna = False)) for j in range(Xc.shape[1])])
			del Xc
			del yc


			#X.flags['WRITEABLE'] = False
			#y.flags['WRITEABLE'] = False

			self.scores_ = res[:,0]
			self.pvalues_ = res[:,1]

		return self

	def _get_support_mask(self):
		if self.k == 'all':
			return np.ones(self.pvalues_.shape, dtype = bool)

		elif self.k == 0:
			return np.zeros(self.pvalues_.shape, dtype = bool)

		else:
			mask = np.zeros(self.pvalues_.shape, dtype = bool)

			# Request a stable sort. Mergesort takes more memory (~40MB per
			# megafeature on x86-64).
			mask[np.argsort(self.pvalues_, kind = "mergesort")[:self.k]] = 1
			return mask

	def get_support(self, indices = False):
		mask = self._get_support_mask()
		return mask if not indices else np.where(mask)[0]

	def transform(self, X):
		mask = self.get_support()
		if not mask.any():
			warn("No features were selected: either the data is"
			" too noisy or the selection test too strict.",
			UserWarning)

			return np.empty(0).reshape((X.shape[0], 0))

		if len(mask) != X.shape[1]:
			raise ValueError("X has a different shape than during fitting.")

		mask = np.asarray(mask)

		if hasattr(X, "toarray"):
			ind = np.arange(mask.shape[0])
			mask = ind[mask]

		return X[:, mask]

	def fit_transform(self, X, y, **fit_params):
		self.set_params(**fit_params)
		return self.fit(X, y).transform(X)
