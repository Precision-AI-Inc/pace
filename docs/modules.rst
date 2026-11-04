Module Reference
================

API documentation for the ``precisionai.pace`` package. Everything listed in
``precisionai.pace.__all__`` is importable from the package itself.

precisionai.pace.core
---------------------

The logger: levels, configuration and output.

.. automodule:: precisionai.pace.core
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.scopes
-----------------------

Progress: ``progress``, ``track``, ``timed``, ``phase`` and ``plan``.

.. automodule:: precisionai.pace.scopes
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.capture
------------------------

Routing the standard library's ``logging`` through Pace.

.. automodule:: precisionai.pace.capture
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.protocol
-------------------------

Progress records and the ``[pace ...]`` trailer: writing, parsing, line splitting.

.. automodule:: precisionai.pace.protocol
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.forecast
-------------------------

Remaining-time estimates for one scope and for a whole job.

.. automodule:: precisionai.pace.forecast
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.lightning
--------------------------

PyTorch Lightning callback (extra: ``precisionai-pace[lightning]``).

.. automodule:: precisionai.pace.lightning
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.transformers
-----------------------------

Hugging Face Transformers and TRL callback (extra: ``precisionai-pace[transformers]``).

.. automodule:: precisionai.pace.transformers
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.tqdm
---------------------

A stand-in for the ``tqdm`` class, for libraries that take a progress-bar class
(no ``tqdm`` install needed).

.. automodule:: precisionai.pace.tqdm
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.pace.version
------------------------

Installed package version.

.. automodule:: precisionai.pace.version
   :members:
   :undoc-members:
   :show-inheritance:
