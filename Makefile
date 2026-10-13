PY = uv run python

.PHONY: data features train predict test

data:
	$(PY) -m gaia_variable_stars.dataset

features:
	$(PY) -m gaia_variable_stars.features

train:
	$(PY) -m gaia_variable_stars.modeling.train

predict:
	$(PY) -m gaia_variable_stars.modeling.predict

test:
	uv run pytest -q