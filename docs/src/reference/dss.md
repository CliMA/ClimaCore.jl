# DSS

```@meta
CurrentModule = ClimaCore
```

Direct stiffness summation makes a field on a continuous spectral-element
space single-valued at element boundaries ([DSS and numerical
fluxes](../explanation/interelement.md)). `Spaces.weighted_dss!` is the
user-level entry point; the `Topologies` functions are the phases it is built
from.

```@docs
Topologies.dss_transform
Topologies.dss_transform!
Topologies.dss_untransform!
Topologies.dss_untransform
Topologies.dss_local!
Topologies.dss_local_ghost!
Topologies.dss_ghost!
Topologies.create_dss_buffer
Topologies.fill_send_buffer!
Topologies.DSSBuffer
Topologies.load_from_recv_buffer!
Topologies.dss!
Spaces.weighted_dss_start!
Spaces.weighted_dss_internal!
Spaces.weighted_dss_ghost!
Spaces.weighted_dss!
Spaces.unique_nodes
```
