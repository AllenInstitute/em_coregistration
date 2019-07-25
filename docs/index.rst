.. em_coregistration documentation master file, created by
   sphinx-quickstart on Tue Jul  9 10:53:15 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to em_coregistration's documentation!
=============================================

This repo was created in the spring of 2019 to coregister optical data from Baylor with EM data from Allen for the IARPA mouse.

3 main functions of this code:

* handle the various coordinate systems, units, and data formats used by Baylor, Allen, Princeton, and APL. This functionality is not comprehensive, but works for the purpose.
* solve for a 3D transform from optical to EM or vice-versa. The transform is a 3D thin-plate spline, though linear and polynomial kernels are also present in the code.
* trim one dataset with the bounds of another. Baylor's optical data covered a larger volume than the fine-aligned EM data (at this time) and we wanted to trim down the optical volume to narrow down to what we might find in the EM.

User Guide
----------

.. toctree::
   :maxdepth: 2

   user_guide

API
---

.. toctree::
   :maxdepth: 2

   api/solve
   api/transform
   api/data_handler
   api/data_filter


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
