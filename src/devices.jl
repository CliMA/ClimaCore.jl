import ClimaComms

# Sometimes, it is convenient to know whether something is running on a device or on its
# host. We define new AbstractDevices and AbstractCommsContext to identify the case of a
# function running on the device with data on the device.

"""
    DeviceSideDevice()

This device represents data defined on the device side of an accelerator.

The most common example is data defined on a GPU. DeviceSideDevice() is used for
operations within the accelerator.
"""
struct DeviceSideDevice <: ClimaComms.AbstractDevice end


"""
    DeviceSideContext()

Context associated to data defined on the device side of an accelerator.

The most common example is data defined on a GPU. DeviceSideContext() is used for
operations within the accelerator.
"""
struct DeviceSideContext <: ClimaComms.AbstractCommsContext end

ClimaComms.context(::DeviceSideDevice) = DeviceSideContext()
ClimaComms.device(::DeviceSideContext) = DeviceSideDevice()
