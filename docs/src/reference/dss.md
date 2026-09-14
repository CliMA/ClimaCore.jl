# DSS

```@meta
CurrentModule = ClimaCore
```

Direct stiffness summation makes a field on a continuous spectral-element
space single-valued at element boundaries ([DSS and numerical
fluxes](../explanation/interelement.md)). `Spaces.weighted_dss!` is the
user-level entry point; the `Topologies` functions are the phases it is built
from, for callers that need to overlap communication with computation or to
run DSS on a data layout directly.

```@docs
Spaces.weighted_dss!
Spaces.weighted_dss_start!
Spaces.weighted_dss_internal!
Spaces.weighted_dss_ghost!
Spaces.unique_nodes
```

## Buffers

```@docs
Topologies.DSSBuffer
Topologies.create_dss_buffer
Topologies.Perimeter2D
```

## Phases

```@docs
Topologies.dss!
Topologies.dss_load_perimeter_data!
Topologies.dss_local!
Topologies.dss_local_ghost!
Topologies.fill_send_buffer!
Topologies.load_from_recv_buffer!
Topologies.dss_ghost!
Topologies.dss_unload_perimeter_data!
```
